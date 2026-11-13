#!/usr/bin/env python3
"""Numerical parity: official ONNX bundle graphs vs transformers checkpoint.

Checks
  1. mel columns (bundle pads to 3000; HF keeps T=L//160)
  2. audio tower hidden rows (ONNX audio_hidden[valid] vs HF mlp_tower(encoder(x)))
  3. projector output == prompt audio embeddings
  4. LM prefill last-position logits (ONNX lm_cache_prefill vs HF Qwen2 stack)
  5. RoPE probe: shift cache_position and see whether logits move
  6. lm_logits.onnx (no-cache graph) vs lm_cache_prefill
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("downloaded/onnx_release").resolve()))
import onnxruntime as ort  # noqa: E402
from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402

BUNDLE = Path("downloaded/onnx_release/model_bundle")


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    d = np.abs(a - b)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_max": float(d.max() / (np.abs(b).max() + 1e-30)), "cosine": cos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--tensor-dir", default="ref/tensors")
    ap.add_argument("--out", default="ref/parity.json")
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    audio_s = ort.InferenceSession(str(BUNDLE / "audio_hidden.onnx"), sess_options=so,
                                   providers=["CPUExecutionProvider"])
    pre_s = ort.InferenceSession(str(BUNDLE / "lm_cache_prefill.onnx"), sess_options=so,
                                 providers=["CPUExecutionProvider"])
    logits_s = ort.InferenceSession(str(BUNDLE / "lm_logits.onnx"), sess_options=so,
                                    providers=["CPUExecutionProvider"])
    dec_s = ort.InferenceSession(str(BUNDLE / "lm_cache_decode.onnx"), sess_options=so,
                                 providers=["CPUExecutionProvider"])
    proj = np.load(BUNDLE / "weights/audio_projector.npz")
    emb = np.load(BUNDLE / "weights/token_embedding.npy", mmap_mode="r")

    processor = AutoProcessor.from_pretrained("downloaded/hf_model", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("downloaded/hf_model", trust_remote_code=True,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    model.language_model.lm_head.weight = model.language_model.model.embed_tokens.weight
    # restore the non-persistent sinusoidal positional buffer (zero-filled on load)
    pe = model.audio_encoder.positional_embedding.positional_embedding
    ch = pe.shape[1]
    inc = np.log(10000) / (ch // 2 - 1)
    inv = np.exp(-inc * np.arange(ch // 2))
    st = np.arange(pe.shape[0])[:, None] * inv[None, :]
    pe.copy_(torch.from_numpy(np.concatenate([np.sin(st), np.cos(st)], axis=1)).to(pe.dtype))

    report = {}
    for stem in args.stems:
        z = np.load(Path(args.tensor_dir) / f"{stem}.npz")
        mel = z["mel"]                      # [1,128,3000] float32
        ids = z["prompt_ids"]
        onnx_hidden = z["audio_hidden"]     # [390,1024]
        mask = z["valid_mask"].astype(bool)
        n_tok = int((ids == 151646).sum())

        # HF only ever saw T = L//160 mel frames; recover the unpadded slice
        # (the bundle pads the tail with exact zeros).
        T = int(mel.shape[2])
        # mel columns beyond the real audio are exactly 0 (constant pad)
        nz = np.where(np.abs(mel[0]).sum(axis=0) > 0)[0]
        T_real = int(nz.max() + 1) if nz.size else T
        feats_hf = torch.from_numpy(mel[0][:, :T_real].copy())

        # ---- 1/2: audio tower parity -------------------------------------
        with torch.inference_mode():
            # audio_encoder.forward expects [mel_bins, frames] (see
            # qwen3_asr_audio_model.py:185 `input_features.T.split(...)`)
            enc = model.audio_encoder(feats_hf,
                                      feature_lens=torch.tensor([T_real], dtype=torch.long))
            tower = model.audio_mlp_tower(enc.last_hidden_state.float())
        hf_hidden = tower.numpy()
        onnx_valid = onnx_hidden[mask]
        res = {"mel_frames_real": T_real, "valid_rows": int(mask.sum()),
               "hf_tower_rows": int(hf_hidden.shape[0])}
        n = min(hf_hidden.shape[0], onnx_valid.shape[0])
        res["audio_hidden"] = stats(onnx_valid[:n], hf_hidden[:n])
        res["valid_mask_prefix_is_true"] = bool(mask[: n].all())
        res["valid_mask_tail_is_false"] = bool(not mask[n:].any())
        res["invalid_rows_are_zero"] = bool(np.abs(onnx_hidden[mask == False]).max() == 0.0)

        # ---- 3: projector parity ----------------------------------------
        x = torch.from_numpy(onnx_valid[:n].copy())
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            x.transpose(0, 1).float().unsqueeze(0), output_size=n_tok).squeeze(0).transpose(0, 1)
        pn = pooled.numpy()
        ln = (pn - pn.mean(-1, keepdims=True)) / np.sqrt(pn.var(-1, keepdims=True) + 1e-5)
        proj_onnx = ln * proj["norm_weight"] + proj["norm_bias"]
        proj_onnx = proj_onnx @ proj["linear_weight"].T + proj["linear_bias"]
        with torch.inference_mode():
            hf_t = torch.from_numpy(hf_hidden)
            hf_pooled = torch.nn.functional.adaptive_avg_pool1d(
                hf_t.transpose(0, 1).float().unsqueeze(0), output_size=n_tok).squeeze(0).transpose(0, 1)
            hf_proj = model.audio_projector(hf_pooled).numpy()
        res["projector"] = stats(proj_onnx, hf_proj)
        res["projector_vs_bundle_embeddings"] = stats(proj_onnx, z["audio_embeddings"])

        # ---- 4: LM prefill parity ----------------------------------------
        e = np.asarray(emb[np.asarray(ids, dtype=np.int64)], dtype=np.float32)
        e[np.where(ids == 151646)[0]] = z["audio_embeddings"]
        L = e.shape[0]
        logits_pre = pre_s.run(None, {"inputs_embeds": e[None], "cache_position": np.arange(L)})[0][0, -1]
        with torch.inference_mode():
            out = model.language_model.model(
                input_ids=None,
                attention_mask=torch.ones((1, L), dtype=torch.long),
                past_key_values=None,
                inputs_embeds=torch.from_numpy(e).unsqueeze(0),
                use_cache=False,
            )
            logits_hf = model.language_model.lm_head(out.last_hidden_state[:, -1:, :])[0, -1].numpy()
        res["prefill_logits"] = stats(logits_pre, logits_hf)
        res["prefill_top1_bundle"] = int(np.argmax(logits_pre))
        res["prefill_top1_hf"] = int(np.argmax(logits_hf))

        # ---- 5: RoPE probe -----------------------------------------------
        logits_shift = pre_s.run(None, {"inputs_embeds": e[None],
                                        "cache_position": np.arange(L) + 64})[0][0, -1]
        res["rope_probe_shift64"] = stats(logits_shift, logits_pre)
        res["rope_probe_top1_changed"] = bool(np.argmax(logits_shift) != np.argmax(logits_pre))
        zero_pos = pre_s.run(None, {"inputs_embeds": e[None],
                                    "cache_position": np.zeros(L, dtype=np.int64)})[0][0, -1]
        res["rope_probe_all_zero_positions"] = stats(zero_pos, logits_pre)

        # ---- 6: no-cache graph -------------------------------------------
        logits_full = logits_s.run(None, {"inputs_embeds": e[None],
                                          "attention_mask": np.ones((1, L), dtype=np.int64)})[0]
        logits_full = logits_full.reshape(-1)
        res["lm_logits_vs_prefill"] = stats(logits_full, logits_pre)

        # ---- 7: decode-graph parity (teacher forcing on the HF greedy path) ----
        caches = [np.zeros((1, 8, 512, 64), dtype=np.float32) for _ in range(16)]
        pre = pre_s.run(None, {"inputs_embeds": e[None], "cache_position": np.arange(L)})
        for i in range(8):
            caches[2 * i][:, :, :L, :] = pre[1 + 2 * i]
            caches[2 * i + 1][:, :, :L, :] = pre[2 + 2 * i]
        # teacher-force the bundle's own greedy tokens through the decode graph
        # and compare each step with a full re-prefill (lm_cache_prefill).
        step_logits = logits_pre
        pos = L
        dec = []
        for _ in range(4):
            tok = int(np.argmax(step_logits))
            emb_tok = np.asarray(emb[np.asarray([tok], dtype=np.int64)], dtype=np.float32)
            am = np.zeros((1, 512), dtype=np.int64)
            am[:, : pos + 1] = 1
            feeds = {"inputs_embeds": emb_tok.reshape(1, 1, -1), "attention_mask": am,
                     "cache_position": np.asarray([pos], dtype=np.int64)}
            for i in range(8):
                feeds[f"cache_key_{i}"] = caches[2 * i]
                feeds[f"cache_value_{i}"] = caches[2 * i + 1]
            outs = dec_s.run(None, feeds)
            step_logits = outs[0][0, -1].astype(np.float32)
            for i in range(8):
                caches[2 * i][:, :, pos:pos + 1, :] = outs[1 + 2 * i]
                caches[2 * i + 1][:, :, pos:pos + 1, :] = outs[2 + 2 * i]
            e = np.concatenate([e, emb_tok], axis=0)
            pos += 1
            ref = pre_s.run(None, {"inputs_embeds": e[None],
                                   "cache_position": np.arange(pos)})[0][0, -1]
            dec.append({"step": len(dec), "token": tok, **stats(step_logits, ref),
                        "top1_decode": int(np.argmax(step_logits)),
                        "top1_ref": int(np.argmax(ref))})
        res["decode_graph_vs_prefill"] = dec

        report[stem] = res
        print(stem, json.dumps({k: (v if not isinstance(v, dict) else
                                    {kk: round(vv, 8) if isinstance(vv, float) else vv
                                     for kk, vv in v.items()}) for k, v in res.items()}, indent=2))

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
