"""Check the manual Qwen2 decoder against the transformers forward pass."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from export_audio8_onnx import AUDIO_TOKEN_ID, MAX_TOTAL_LEN, DecoderExport

ROOT = Path("/Users/yunyu/Documents/DSH/AX-model/audio8")
MODEL_DIR = str(ROOT / "downloaded/hf")


def main() -> None:
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager").eval()
    lm = model.language_model
    lm.lm_head.weight = lm.model.embed_tokens.weight

    wav = str(ROOT / "wavs/1.wav")
    conversation = [{"role": "user", "content": [
        {"type": "audio", "path": wav}, {"type": "text", "text": "Please transcribe this audio."}]}]
    batch = processor.apply_chat_template(
        conversation, return_tensors="pt", sampling_rate=16000,
        audio_padding="longest", add_generation_prompt=True,
        audio_max_length=30 * 16000,
        text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000})
    input_ids = batch["input_ids"]
    features = batch["input_features"]
    n = int((input_ids[0] == AUDIO_TOKEN_ID).sum())
    print("prompt_len", input_ids.shape[1], "audio_tokens", n, flush=True)

    # --- HF reference logits for the last position
    out = model(input_ids=input_ids, input_features=features, use_cache=True)
    ref = out.logits[0, -1, :]

    # --- our decoder: encoder hidden first
    from export_audio8_onnx import EncoderExport
    enc = EncoderExport(model).eval()
    feats = features.float()
    flen = torch.tensor([feats.shape[-1]], dtype=torch.int64)
    hidden, mask = enc(feats, flen)
    valid = hidden[0, : int(mask[0].sum())]
    # adaptive average pool to n tokens
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        valid.transpose(0, 1).unsqueeze(0), n).squeeze(0).transpose(0, 1)

    dec = DecoderExport(model, MAX_TOTAL_LEN).eval()
    t = input_ids.shape[1]
    attn = torch.zeros(1, MAX_TOTAL_LEN, dtype=torch.int64)
    attn[0, :t] = 1
    pos = torch.arange(t, dtype=torch.int64)
    caches = []
    for _ in range(dec.num_layers):
        caches.append(torch.zeros(1, dec.num_kv_heads, MAX_TOTAL_LEN, dec.head_dim))
        caches.append(torch.zeros(1, dec.num_kv_heads, MAX_TOTAL_LEN, dec.head_dim))
    outs = dec(input_ids, pooled.unsqueeze(0), attn, pos, *caches)
    ours = outs[0][0, -1, :]

    diff = (ref - ours).abs()
    cos = torch.nn.functional.cosine_similarity(ref, ours, dim=0)
    print(f"logits: max|diff|={diff.max().item():.6f} mean|diff|={diff.mean().item():.6f} "
          f"cosine={cos.item():.8f}")
    print("argmax ref/ours:", int(ref.argmax()), int(ours.argmax()), flush=True)

    info = {"prompt_len": t, "audio_tokens": n,
            "max_abs_diff": float(diff.max()), "cosine": float(cos),
            "argmax_ref": int(ref.argmax()), "argmax_ours": int(ours.argmax())}
    (ROOT / "reports/decoder-equiv.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
