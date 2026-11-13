"""Load Audio8-ASR-0.1B with transformers and transcribe test wavs (CPU, fp32).

This is the ground truth for the ONNX export: same weights, same greedy decode.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path("/Users/yunyu/Documents/DSH/AX-model/audio8")
MODEL = str(ROOT / "downloaded/hf")


PAD = "--no-pad" not in sys.argv
sys.argv = [a for a in sys.argv if a != "--no-pad"]


def main() -> None:
    wavs = sys.argv[1:] or [str(ROOT / "wavs/0.wav"), str(ROOT / "wavs/3.wav")]
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    # The release drops the tied `lm_head` tensor, so transformers leaves it
    # randomly initialised. Tie it back to the token embedding explicitly.
    lm = model.language_model
    if getattr(lm.lm_head.weight, "data_ptr", None) is None or True:
        lm.lm_head.weight = lm.model.embed_tokens.weight
        print("[ref] tied lm_head -> embed_tokens", flush=True)
        import numpy as np
        pe = model.audio_encoder.positional_embedding.positional_embedding
        ch = pe.shape[1]
        inc = np.log(10000) / (ch // 2 - 1)
        inv = np.exp(-inc * np.arange(ch // 2))
        scaled = np.arange(pe.shape[0])[:, None] * inv[None, :]
        table = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
        import torch as _t
        with _t.no_grad():
            pe.copy_(_t.from_numpy(table).to(pe.dtype))
        print("[ref] restored sinusoid positions", flush=True)

    results = []
    for wav in wavs:
        conversation = [{
            "role": "user",
            "content": [{"type": "audio", "path": wav},
                        {"type": "text", "text": "Please transcribe this audio."}],
        }]
        batch = processor.apply_chat_template(
            conversation, return_tensors="pt", sampling_rate=16000,
            audio_padding="longest", add_generation_prompt=True,
            audio_max_length=30 * 16000,
            text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
        )
        # the official runtime pads the mel to the full 30 s window (3000 frames);
        # pad here too so the reference matches that pipeline
        feats = batch["input_features"]
        if PAD and feats.shape[-1] < 3000:
            import torch.nn.functional as F
            batch["input_features"] = F.pad(feats, (0, 3000 - feats.shape[-1]))
        t0 = time.perf_counter()
        out = model.generate(**batch, max_new_tokens=128, do_sample=False)
        dt = time.perf_counter() - t0
        prompt_len = int(batch["input_ids"].shape[1])
        text = processor.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
        ids = out[0, prompt_len:].tolist()
        entry = {"wav": Path(wav).name, "text": text, "generated_ids": ids,
                 "prompt_len": prompt_len, "seconds": round(dt, 2)}
        # extra: prompt ids and audio token count, needed by the C++ runtime
        entry["prompt_ids"] = batch["input_ids"][0].tolist()
        entry["audio_token_count"] = int((batch["input_ids"][0] == 151646).sum())
        if "input_features" in batch:
            entry["input_features_shape"] = list(batch["input_features"].shape)
        results.append(entry)
        print(json.dumps({k: entry[k] for k in ("wav", "text", "prompt_len",
                                               "audio_token_count", "seconds")},
                         ensure_ascii=False), flush=True)

    (ROOT / "reports").mkdir(exist_ok=True)
    out_json = "hf-reference.json" if PAD else "hf-reference-unpadded.json"
    (ROOT / "reports" / out_json).write_text(
        json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
