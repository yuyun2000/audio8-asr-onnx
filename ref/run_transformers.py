#!/usr/bin/env python3
"""Transformers reference run for Edge0/Audio8-ASR-0.1B (arch "arkasr").

Mirrors examples/transcribe.py from the HF repo, but also dumps the exact
prompt token ids, the mel features actually fed to the audio tower, the
encoder output length and the per-utterance audio-token count.

Usage:
  python ref/run_transformers.py downloaded/wavs/0.wav [...] --out ref/hf_runs.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = "Please transcribe this audio."


def build_conversation(audio_path: Path) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": str(audio_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def audio_token_count(mel_frames: int, merge_factor: int = 4) -> int:
    """ArkasrProcessor.calculate_audio_token_count (processing_arkasr.py:162)."""
    downsampled = (int(mel_frames) + 1) // 2
    merged = downsampled // max(int(merge_factor), 1)
    return max(int(merged), 1)


def enc_output_len(feature_len: int) -> int:
    """qwen3_asr_audio_model._get_feat_extract_output_lengths (line 17)."""
    leave = feature_len % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (feature_len // 100) * 13


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", type=Path)
    ap.add_argument("--model", default="downloaded/hf_model")
    ap.add_argument("--out", default="ref/hf_runs.json")
    ap.add_argument("--tensor-dir", default="ref/tensors_hf")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--audio-max-seconds", type=int, default=30)
    ap.add_argument("--audio-dtype", default=None,
                    help="override processor audio_dtype, e.g. float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-restore-pos-emb", dest="restore_pos", action="store_false", default=True)
    args = ap.parse_args()

    tensor_dir = Path(args.tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    load_t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if args.audio_dtype:
        processor.audio_dtype = args.audio_dtype
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    # transformers 5.22.2 zero-fills the non-persistent `positional_embedding`
    # buffer of SinusoidsPositionEmbedding (qwen3_asr_audio_model.py:107-122).
    # It is not stored in the checkpoint, so any loader MUST recompute it;
    # without it the audio tower silently loses positional information.
    if args.restore_pos:
        pe = model.audio_encoder.positional_embedding.positional_embedding
        ch = pe.shape[1]
        inc = np.log(10000) / (ch // 2 - 1)
        inv = np.exp(-inc * np.arange(ch // 2))
        st = np.arange(pe.shape[0])[:, None] * inv[None, :]
        pe.copy_(torch.from_numpy(np.concatenate([np.sin(st), np.cos(st)], axis=1)).to(pe.dtype))
        print(f"[pos-emb] restored sinusoid buffer {tuple(pe.shape)} max={pe.abs().max().item():.4f}")

    # transformers 5.22.2 does NOT honor `_tied_weights_keys` in the list form used
    # by the checkpoint's custom code (checkpoint was written with transformers
    # 4.57.3). Result: `language_model.lm_head.weight` is reported MISSING and
    # freshly initialized with N(0, 0.02). The config sets
    # tie_word_embeddings=True and the checkpoint contains no lm_head tensor, so
    # the lm_head MUST be the embedding matrix. Re-tie it explicitly.
    emb = model.language_model.model.embed_tokens.weight
    model.language_model.lm_head.weight = emb
    tied_ok = model.language_model.lm_head.weight.data_ptr() == emb.data_ptr()
    print(f"[tie] lm_head tied to embed_tokens: {tied_ok}")

    load_s = time.perf_counter() - load_t0
    fe = processor.feature_extractor
    meta = {
        "model": args.model,
        "load_seconds": load_s,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "merge_factor": processor.merge_factor,
        "audio_dtype_used": processor.audio_dtype,
        "feature_extractor": {
            "n_fft": fe.n_fft, "hop_length": fe.hop_length,
            "feature_size": fe.feature_size, "nb_max_frames": fe.nb_max_frames,
            "n_samples": fe.n_samples, "sampling_rate": fe.sampling_rate,
        },
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "lm_head_retied": True,
        "pos_emb_restored": bool(args.restore_pos),
        "tie_note": "transformers 5.22.2 ignores _tied_weights_keys list form; lm_head re-tied to embed_tokens",
    }
    print(json.dumps(meta, indent=2))

    runs = []
    for wav in args.wavs:
        conv = build_conversation(wav)
        batch = processor.apply_chat_template(
            conv,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
            add_generation_prompt=True,
            audio_max_length=int(args.audio_max_seconds) * 16000,
            text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
        )
        batch = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in dict(batch).items()}
        prompt_ids = batch["input_ids"][0].tolist()
        feats = batch["input_features"][0].float().numpy()  # [128, T]
        n_audio_tok = sum(1 for t in prompt_ids if t == 151646)
        mel_frames = int(feats.shape[1])

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen_s = time.perf_counter() - t0
        prompt_len = int(batch["input_ids"].shape[1])
        new_ids = out[0, prompt_len:].tolist()
        text = processor.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

        stem = wav.stem
        np.savez_compressed(
            tensor_dir / f"{stem}.npz",
            prompt_ids=np.asarray(prompt_ids, dtype=np.int64),
            input_features=feats.astype(np.float32),
            generated_ids=np.asarray(new_ids, dtype=np.int64),
        )
        run = {
            "wav": str(wav),
            "wav_samples": int(round(mel_frames * 160)),
            "mel_frames": mel_frames,
            "mel_shape": list(feats.shape),
            "prompt_len": prompt_len,
            "audio_token_count": n_audio_tok,
            "audio_token_count_rule": audio_token_count(mel_frames, processor.merge_factor),
            "encoder_output_len": enc_output_len(mel_frames),
            "prompt_ids_head": prompt_ids[:3],
            "prompt_ids_tail": prompt_ids[-8:],
            "prompt_text_tokens": [processor.tokenizer.convert_ids_to_tokens(t) for t in prompt_ids[2 + n_audio_tok: -1]],
            "generated_ids": new_ids,
            "generated_tokens": len(new_ids),
            "hit_eos": bool(new_ids and new_ids[-1] in (151645, 151643)),
            "text": text,
            "seconds": gen_s,
            "seconds_per_audio_second": gen_s / max(mel_frames / 100.0, 1e-9),
        }
        runs.append(run)
        print(json.dumps({k: v for k, v in run.items() if k != "generated_ids"}, ensure_ascii=False, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
