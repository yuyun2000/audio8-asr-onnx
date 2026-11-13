"""Locate the mismatch between the manual decoder and the HF forward pass."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from export_audio8_onnx import AUDIO_TOKEN_ID

ROOT = __import__("pathlib").Path("/Users/yunyu/Documents/DSH/AX-model/audio8")
MODEL_DIR = str(ROOT / "downloaded/hf")


def main() -> None:
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager").eval()
    lm = model.language_model
    lm.lm_head.weight = lm.model.embed_tokens.weight

    wav = str(ROOT / "wavs/3.wav")
    conv = [{"role": "user", "content": [
        {"type": "audio", "path": wav}, {"type": "text", "text": "Please transcribe this audio."}]}]
    batch = processor.apply_chat_template(
        conv, return_tensors="pt", sampling_rate=16000, audio_padding="longest",
        add_generation_prompt=True, audio_max_length=30 * 16000,
        text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000})
    ids = batch["input_ids"]
    feats = batch["input_features"].float()
    n_audio = int((ids[0] == AUDIO_TOKEN_ID).sum())

    # --- HF embeddings
    hf_embed = lm.model.embed_tokens(ids)
    audio_hidden = model._project_audio_row(feats[0], n_audio, torch.float32)
    positions = torch.nonzero(ids[0] == AUDIO_TOKEN_ID).flatten()
    hf_embed[0, positions, :] = audio_hidden
    out = model(input_ids=ids, input_features=feats, output_hidden_states=True, use_cache=True)
    hf_hidden = out.hidden_states[0]  # embeddings after injection (layer 0 input)

    # --- our inputs
    from export_audio8_onnx import EncoderExport
    enc = EncoderExport(model).eval()
    hidden, mask = enc(feats, torch.tensor([feats.shape[-1]], dtype=torch.int64))
    valid = hidden[0, : int(mask[0].sum())]
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        valid.transpose(0, 1).unsqueeze(0), n_audio).squeeze(0).transpose(0, 1)
    ours = model.audio_projector(pooled)

    print("audio projection diff:", (ours - audio_hidden).abs().max().item())
    print("embedding diff:", (hf_embed - hf_embed).abs().max().item())

    # compare the hidden states after each layer using our layer-wise path
    h = hf_embed
    t = ids.shape[1]
    pos = torch.arange(t)
    attn = torch.ones(1, t)
    for i, layer in enumerate(lm.model.layers):
        h_hf = out.hidden_states[i + 1]
        # our layer
        x = layer.input_layernorm(h)
        q = layer.self_attn.q_proj(x).view(1, t, 8, 64).transpose(1, 2)
        k = layer.self_attn.k_proj(x).view(1, t, 8, 64).transpose(1, 2)
        v = layer.self_attn.v_proj(x).view(1, t, 8, 64).transpose(1, 2)
        import math
        cos = torch.cos(pos[:, None] * torch.tensor(1.0) * 0.0)
        # use HF rotary module
        rot = lm.model.rotary_emb
        cos_s, sin_s = rot(x, pos[None, :])
        def apply(t_, name):
            c = cos_s[0][None, None, :, :]
            s = sin_s[0][None, None, :, :]
            t1, t2 = t_[..., :32], t_[..., 32:]
            rot_half = torch.cat([-t2, t1], dim=-1)
            return t_ * c + rot_half * s
        q = apply(q, "q")
        k = apply(k, "k")
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(64)
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
        scores = scores.masked_fill(~causal[None, None], torch.finfo(torch.float32).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        o = torch.matmul(probs, v).transpose(1, 2).reshape(1, t, 512)
        h = h + layer.self_attn.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        print(f"layer {i}: hidden diff {(h - h_hf).abs().max().item():.6f} "
              f"cosine {torch.nn.functional.cosine_similarity(h.flatten(), h_hf.flatten(), dim=0).item():.6f}")


if __name__ == "__main__":
    main()
