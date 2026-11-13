#!/usr/bin/env python3
"""Reference runner for the official Audio8-ASR ONNX bundle (fp32 graphs).

Wraps `asr_onnx_runtime.OnnxCacheAsrEngine` unchanged, but captures the exact
intermediates the runtime feeds the graphs (mel, prompt token ids, audio
embeddings) and re-runs the audio graph once to record its raw IO.

Usage:
  python ref/run_bundle.py ref/transcripts.json downloaded/wavs/0.wav ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RELEASE = Path("downloaded/onnx_release")
BUNDLE = RELEASE / "model_bundle"
sys.path.insert(0, str(RELEASE.resolve()))

import onnxruntime as ort  # noqa: E402
from asr_onnx_runtime import OnnxCacheAsrEngine, OnnxAsrEngine, ark_audio_token_count  # noqa: E402


class CaptureEngine(OnnxCacheAsrEngine):
    """Official engine + capture of every tensor that crosses a graph boundary."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cap: dict = {}

    def _extract_features(self, audio):
        out = super()._extract_features(audio)
        feature, sample_count, encoder_feature_len, hop_length = out
        self.cap.update(
            audio=audio,
            mel=feature,
            sample_count=int(sample_count),
            encoder_feature_len=int(encoder_feature_len),
            hop_length=int(hop_length),
        )
        return out

    def _audio_embeddings(self, feature, sample_count, encoder_feature_len, hop_length):
        emb = super()._audio_embeddings(feature, sample_count, encoder_feature_len, hop_length)
        hidden, valid_mask = self.audio_session.run(
            None,
            {
                "audios": feature.astype(np.float32, copy=False),
                "audio_feature_lengths": np.asarray([encoder_feature_len], dtype=np.int64),
            },
        )
        self.cap.update(
            audio_hidden=hidden,
            valid_mask=valid_mask,
            audio_embeddings=emb,
            runtime_token_count=int(ark_audio_token_count(
                sample_count, hop_length=hop_length, merge_factor=self.prompt_merge_factor)),
        )
        return emb

    def _initial_embeddings(self, audio_embeddings, language):
        ids, embeds = super()._initial_embeddings(audio_embeddings, language=language)
        self.cap.update(prompt_ids=np.asarray(ids, dtype=np.int64), inputs_embeds=embeds,
                        prompt_str=self._build_prompt(int(audio_embeddings.shape[0])))
        return ids, embeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref/transcripts.json")
    ap.add_argument("--hf-runs", default="ref/hf_runs.json")
    ap.add_argument("--tensor-dir", default="ref/tensors")
    ap.add_argument("--cache-precision", default="fp32")
    ap.add_argument("--audio-precision", default="fp32")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--engine", default="cache", choices=["cache", "full"], help="cache=lm_cache_* graphs, full=lm_logits")
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    tensor_dir = Path(args.tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    cls = CaptureEngine if args.engine == "cache" else type(
        "CaptureFull", (OnnxAsrEngine, CaptureEngine), {}
    )
    t0 = time.perf_counter()
    engine = cls(BUNDLE, provider="CPUExecutionProvider", cache_precision=args.cache_precision,
                 audio_precision=args.audio_precision)
    load_s = time.perf_counter() - t0
    print(f"engine={cls.__name__} load_seconds={load_s:.2f} providers={engine.providers}")
    if getattr(engine, "lm_cache_dtype", None) is not None:
        print(f"lm cache dtype={engine.lm_cache_dtype} embed dtype={engine.lm_embed_dtype} "
              f"cache={engine._new_cache()[0].shape}")

    runs = []
    for wav in args.wavs:
        wav = Path(wav)
        data = wav.read_bytes()
        res = engine.transcribe(data, max_new_tokens=args.max_new_tokens, do_sample=False)
        cap = engine.cap
        stem = wav.stem
        np.savez_compressed(
            tensor_dir / f"{stem}.npz",
            mel=cap["mel"].astype(np.float32),
            prompt_ids=cap["prompt_ids"],
            inputs_embeds=cap["inputs_embeds"].astype(np.float32),
            audio_embeddings=cap["audio_embeddings"].astype(np.float32),
            audio_hidden=cap["audio_hidden"].astype(np.float32),
            valid_mask=np.asarray(cap["valid_mask"]).astype(np.int8),
            audio=cap["audio"].astype(np.float32),
        )
        run = {
            "wav": str(wav),
            "bundle_text": res["text"],
            "bundle_raw": res["raw"],
            "generated_tokens": res["generated_tokens"],
            "hit_stop": res["hit_stop"],
            "stop_token_id": res["stop_token_id"],
            "seconds": res["elapsed_seconds"],
            "audio_seconds": res["audio_seconds"],
            "audio_token_count": res["audio_token_count"],
            "runtime_token_count": cap["runtime_token_count"],
            "mel_shape": list(cap["mel"].shape),
            "encoder_feature_len": cap["encoder_feature_len"],
            "sample_count": cap["sample_count"],
            "audio_hidden_shape": list(cap["audio_hidden"].shape),
            "valid_mask_sum": int(np.asarray(cap["valid_mask"]).astype(bool).sum()),
            "prompt_len": int(cap["prompt_ids"].shape[0]),
            "prompt_ids_head": cap["prompt_ids"][:3].tolist(),
            "prompt_ids_tail": cap["prompt_ids"][-8:].tolist(),
            "prompt_str_prefix": cap["prompt_str"][:120],
            "cache_precision": getattr(engine, "cache_precision", None),
            "audio_precision": engine.audio_precision,
        }
        runs.append(run)
        print(json.dumps({k: run[k] for k in ("wav", "bundle_text", "generated_tokens", "hit_stop",
                                              "stop_token_id", "seconds", "audio_seconds",
                                              "audio_token_count", "runtime_token_count",
                                              "mel_shape", "audio_hidden_shape", "valid_mask_sum",
                                              "prompt_len")}, ensure_ascii=False, indent=2))

    # merge transformers reference
    hf = {}
    hp = Path(args.hf_runs)
    if hp.exists():
        hj = json.loads(hp.read_text(encoding="utf-8"))
        for r in hj["runs"]:
            hf[Path(r["wav"]).stem] = r
        hf_meta = hj.get("meta", {})
    else:
        hf_meta = {}

    out = {
        "bundle_dir": str(BUNDLE),
        "engine": cls.__name__,
        "engine_load_seconds": load_s,
        "ort_version": ort.__version__,
        "cache_precision": args.cache_precision,
        "audio_precision": args.audio_precision,
        "max_new_tokens": args.max_new_tokens,
        "runs": [],
    }
    for r in runs:
        stem = Path(r["wav"]).stem
        h = hf.get(stem, {})
        out["runs"].append({
            "wav": r["wav"],
            "bundle_text": r["bundle_text"],
            "transformers_text": h.get("text"),
            "match": (h.get("text") is not None and h.get("text") == r["bundle_text"]),
            "timing": {
                "bundle_seconds": round(r["seconds"], 4),
                "transformers_seconds": round(h.get("seconds", float("nan")), 4) if h else None,
                "audio_seconds": round(r["audio_seconds"], 3),
                "engine_load_seconds": round(load_s, 3),
                "bundle_rtf": round(r["seconds"] / max(r["audio_seconds"], 1e-9), 4),
            },
            "bundle": {k: v for k, v in r.items() if k not in ("wav", "bundle_text", "seconds", "audio_seconds")},
            "transformers": {k: h.get(k) for k in ("prompt_len", "audio_token_count", "encoder_output_len",
                                                   "mel_frames", "generated_tokens", "hit_eos")} if h else None,
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.out)
    print("hf meta:", json.dumps(hf_meta, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
