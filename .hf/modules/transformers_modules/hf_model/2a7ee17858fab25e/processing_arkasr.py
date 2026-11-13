# coding=utf-8
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import librosa
import soundfile as sf  # Used for BytesIO fallback decoding.

from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)

_AUDIO_MARKER = "<<AUDIO_TOKENS>>"

def _normalize_dtype_name(name: str) -> str:
    name = name.strip().lower()
    alias = {
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "float": "float32",
    }
    return alias.get(name, name)


def _resolve_torch_dtype(x: Any, default: str = "float32") -> torch.dtype:
    if isinstance(x, torch.dtype):
        return x
    if x is None:
        x = default
    if isinstance(x, str):
        name = _normalize_dtype_name(x)
        if not hasattr(torch, name):
            raise ValueError(f"Unknown torch dtype string: {x} (normalized: {name})")
        return getattr(torch, name)
    raise TypeError(f"audio_dtype/audio_torch_dtype must be str or torch.dtype or None, got {type(x)}")


class ArkasrProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    valid_kwargs = ["merge_factor", "audio_token", "audio_dtype"]
    feature_extractor_class = ("WhisperFeatureExtractor", "SequenceFeatureExtractor")
    tokenizer_class = ("PreTrainedTokenizerFast", "PreTrainedTokenizer")

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        merge_factor: int = 4,
        audio_token: str = "<|audio|>",
        audio_dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(feature_extractor, tokenizer)
        self.merge_factor = int(merge_factor)
        self.audio_token = str(audio_token)
        self.audio_dtype = str(audio_dtype)

        self.bos_audio_token = "<|begin_of_audio|>"
        self.eos_audio_token = "<|end_of_audio|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.assistant_end_token = "<|im_end|>"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "ArkasrProcessor":
        trust_remote_code = bool(kwargs.pop("trust_remote_code", False))
        passthrough_keys = {"cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder"}
        shared_kwargs = {k: kwargs[k] for k in list(kwargs.keys()) if k in passthrough_keys}

        merge_factor = 4
        audio_token = "<|audio|>"
        audio_dtype = "float32"
        tokenizer_cfg: Dict[str, Any] = {}
        feat_cfg: Dict[str, Any] = {}

        proc_cfg_path = os.path.join(pretrained_model_name_or_path, "processor_config.json")
        if os.path.isfile(proc_cfg_path):
            with open(proc_cfg_path, "r", encoding="utf-8") as f:
                proc_cfg = json.load(f)
            merge_factor = int(proc_cfg.get("merge_factor", merge_factor))
            audio_token = str(proc_cfg.get("audio_token", audio_token))
            audio_dtype = str(proc_cfg.get("audio_dtype", audio_dtype))
            tokenizer_cfg = proc_cfg.get("tokenizer_config", {}) or {}
            feat_cfg = proc_cfg.get("feature_extractor_config", {}) or {}

        feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **shared_kwargs)
        for k, v in feat_cfg.items():
            if hasattr(feature_extractor, k):
                try: setattr(feature_extractor, k, v)
                except Exception: pass

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, use_fast=True, trust_remote_code=trust_remote_code, **shared_kwargs
        )
        for k, v in tokenizer_cfg.items():
            if hasattr(tokenizer, k):
                try: setattr(tokenizer, k, v)
                except Exception: pass

        return cls(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            merge_factor=merge_factor,
            audio_token=audio_token,
            audio_dtype=audio_dtype,
        )

    # =========================
    # Audio helpers.
    # =========================
    def _load_audio_file(self, path: str, sampling_rate: int = 16000, offset: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
        # librosa supports offset and duration for file inputs.
        audio_array, _ = librosa.load(path, sr=int(sampling_rate), mono=True, offset=offset, duration=duration)
        return np.asarray(audio_array, dtype=np.float32)

    def _strip_data_url_prefix(self, b64: str) -> str:
        if "," in b64 and b64[:30].lower().startswith("data:"):
            return b64.split(",", 1)[1]
        return b64

    def _load_audio_base64(self, b64: str, sampling_rate: int = 16000, offset: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
        b64 = self._strip_data_url_prefix(b64)
        raw = base64.b64decode(b64)
        bio = io.BytesIO(raw)
        
        # librosa also supports offset and duration for BytesIO inputs.
        try:
            wav, _sr = librosa.load(bio, sr=int(sampling_rate), mono=True, offset=offset, duration=duration)
            return np.asarray(wav, dtype=np.float32)
        except Exception as e:
            # Fallback path with manual slicing.
            try:
                bio.seek(0)
                data, sr = sf.read(bio, dtype="float32", always_2d=True)
                wav = data.mean(axis=1)
                if int(sr) != int(sampling_rate):
                    wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sampling_rate))
                
                start_sample = int(offset * sampling_rate)
                end_sample = None
                if duration is not None:
                    end_sample = start_sample + int(duration * sampling_rate)
                
                return np.asarray(wav[start_sample:end_sample], dtype=np.float32)
            except Exception as e2:
                 raise ValueError("Failed to decode base64 audio.") from e2

    def calculate_audio_token_count(self, mel_frames: int) -> int:
        downsampled = (int(mel_frames) + 1) // 2
        merged = downsampled // max(self.merge_factor, 1)
        return max(int(merged), 1)

    def _build_templates_and_audios(
        self,
        conversations: List[List[dict]],
        sampling_rate: int,
        add_generation_prompt: bool,
    ) -> tuple[List[str], List[np.ndarray], List[int]]:
        prompts_template: List[str] = []
        audios_raw: List[np.ndarray] = []
        prompt_audio_counts: List[int] = []

        for conv in conversations:
            conv_str = ""
            last_role = None
            audio_count_this_conv = 0

            for msg in conv:
                role = msg["role"]
                last_role = role
                content = msg["content"]

                if role == "user": conv_str += f"{self.user_token}"
                elif role == "assistant": conv_str += f"{self.assistant_token}"
                else: conv_str += f"<|{role}|>"

                if isinstance(content, str):
                    conv_str += f"{content}"
                elif isinstance(content, list):
                    for part in content:
                        ptype = part.get("type")
                        if ptype == "audio":
                            # Optional segment selection from begin_time/end_time.
                            begin_time = part.get("begin_time", -1)
                            end_time = part.get("end_time", -1)
                            
                            offset = 0.0
                            duration = None
                            
                            # Apply slicing only when begin_time is valid.
                            if begin_time is not None and begin_time >= 0:
                                offset = float(begin_time)
                                if end_time is not None and end_time > begin_time:
                                    duration = float(end_time) - float(begin_time)
                            
                            audio_raw_this = None
                            if "array" in part:
                                arr = part["array"]
                                if isinstance(arr, torch.Tensor):
                                    arr = arr.detach().cpu().numpy()
                                full_arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                                
                                # Slice in-memory audio arrays as needed.
                                start_idx = int(offset * sampling_rate)
                                end_idx = None
                                if duration is not None:
                                    end_idx = start_idx + int(duration * sampling_rate)
                                audio_raw_this = full_arr[start_idx:end_idx]

                            elif "path" in part:
                                audio_raw_this = self._load_audio_file(
                                    part["path"], 
                                    sampling_rate=sampling_rate,
                                    offset=offset,
                                    duration=duration
                                )
                            elif "base64" in part:
                                audio_raw_this = self._load_audio_base64(
                                    part["base64"], 
                                    sampling_rate=sampling_rate,
                                    offset=offset,
                                    duration=duration
                                )
                            else:
                                raise ValueError("Audio part must contain 'path' or 'array' or 'base64'.")

                            audios_raw.append(audio_raw_this)
                            audio_count_this_conv += 1
                            conv_str += f"{self.bos_audio_token}{_AUDIO_MARKER}{self.eos_audio_token}"

                        elif ptype == "text":
                            conv_str += f"{part.get('text', '')}"
                        else:
                            raise ValueError(f"Unknown content part type: {ptype}")
                else:
                    raise ValueError(f"Unsupported message content type: {type(content)}")

            if add_generation_prompt:
                if last_role == "user": conv_str += f"{self.assistant_token}"
                elif last_role == "assistant": conv_str += f"{self.assistant_end_token}"
                else: conv_str += f"{self.assistant_token}"

            prompts_template.append(conv_str)
            prompt_audio_counts.append(audio_count_this_conv)

        return prompts_template, audios_raw, prompt_audio_counts

    def _calculate_audio_token_counts_per_sample(
        self,
        audios_raw: List[np.ndarray],
        sampling_rate: int,
        audio_max_length: Optional[int],
        audio_pad_to_multiple_of: Optional[int],
    ) -> List[int]:
        del sampling_rate, audio_pad_to_multiple_of

        hop_length = int(getattr(self.feature_extractor, "hop_length", 160))
        max_audio_samples = int(audio_max_length) if audio_max_length is not None else None
        token_counts: List[int] = []

        for audio_raw in audios_raw:
            audio_np = np.asarray(audio_raw, dtype=np.float32).reshape(-1)
            effective_len = int(audio_np.shape[0])
            if max_audio_samples is not None:
                effective_len = min(effective_len, max_audio_samples)

            mel_frames = effective_len // max(hop_length, 1)
            token_counts.append(self.calculate_audio_token_count(int(mel_frames)))

        return token_counts

    # =========================
    # apply_chat_template
    # =========================
    def apply_chat_template(
        self,
        conversation: Union[List[dict], List[List[dict]]],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[BatchFeature, str, List[str]]:
        if chat_template is not None:
            logger.warning("chat_template argument is ignored.")

        tokenize = kwargs.pop("tokenize", True)
        return_tensors = kwargs.pop("return_tensors", "pt")
        kwargs.pop("return_dict", None)

        audio_torch_dtype = kwargs.pop("audio_torch_dtype", None)
        audio_dtype_override = kwargs.pop("audio_dtype", None)
        dtype_source = audio_torch_dtype if audio_torch_dtype is not None else audio_dtype_override
        target_dtype = _resolve_torch_dtype(dtype_source, default=getattr(self, "audio_dtype", "float32"))

        text_kwargs = dict(kwargs.pop("text_kwargs", {}) or {})
        for k in ("padding", "truncation", "max_length", "add_special_tokens"):
            if k in kwargs and k not in text_kwargs:
                text_kwargs[k] = kwargs.pop(k)

        sampling_rate = int(kwargs.pop("sampling_rate", 16000))
        audio_padding = kwargs.pop("audio_padding", "longest")
        audio_max_length = kwargs.pop("audio_max_length", None)
        audio_pad_to_multiple_of = kwargs.pop("audio_pad_to_multiple_of", None)

        if kwargs:
            logger.warning(f"Ignored unused kwargs: {list(kwargs.keys())}")

        if isinstance(conversation, list) and conversation and isinstance(conversation[0], dict):
            conversations = [conversation]
            is_single = True
        else:
            conversations = conversation
            is_single = False

        prompt_templates, audios_raw, prompt_audio_counts = self._build_templates_and_audios(
            conversations=conversations,
            sampling_rate=sampling_rate,
            add_generation_prompt=add_generation_prompt,
        )

        input_features = None
        audio_token_counts: List[int] = []
        
        if len(audios_raw) > 0:
            feat = self.feature_extractor(
                audios_raw,
                sampling_rate=sampling_rate,
                return_tensors="np",
                return_attention_mask=False,
                padding=audio_padding,
                max_length=audio_max_length,
                pad_to_multiple_of=audio_pad_to_multiple_of,
            )
            input_features = feat["input_features"]
            if not isinstance(input_features, np.ndarray):
                input_features = np.asarray(input_features)

            audio_token_counts = self._calculate_audio_token_counts_per_sample(
                audios_raw=audios_raw,
                sampling_rate=sampling_rate,
                audio_max_length=audio_max_length,
                audio_pad_to_multiple_of=audio_pad_to_multiple_of,
            )

        prompts: List[str] = []
        audio_idx = 0
        for prompt_template, audio_count in zip(prompt_templates, prompt_audio_counts):
            prompt = prompt_template
            for _ in range(audio_count):
                if audio_idx >= len(audio_token_counts):
                    raise ValueError("Audio token count mismatch while building prompts.")
                audio_tokens_str = "".join([self.audio_token] * audio_token_counts[audio_idx])
                prompt = prompt.replace(_AUDIO_MARKER, audio_tokens_str, 1)
                audio_idx += 1
            if _AUDIO_MARKER in prompt:
                raise ValueError("Unresolved audio marker remained in prompt.")
            prompts.append(prompt)

        if audio_idx != len(audio_token_counts):
            raise ValueError("Unused audio token counts remained after prompt construction.")

        if not tokenize:
            return prompts[0] if is_single else prompts

        text_kwargs.setdefault("padding", "longest")
        text_kwargs.setdefault("add_special_tokens", False)
        text_kwargs["return_tensors"] = return_tensors

        enc = self.tokenizer(prompts, **text_kwargs)
        data: Dict[str, Any] = dict(enc)

        if input_features is not None:
            audio_tensor = torch.tensor(input_features, dtype=target_dtype)
            data["input_features"] = audio_tensor
            data["audios"] = audio_tensor

        return BatchFeature(data=data, tensor_type=return_tensors)
    
    # Tokenizer proxy methods and direct audio-array path.
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def __call__(
        self,
        text: Union[str, List[str]],
        audios: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        sampling_rate: int = 16000,
        return_tensors: str = "pt",
        **tokenizer_kwargs,
    ) -> BatchFeature:
        # Direct audio-array path; segment slicing is handled by apply_chat_template.
        audios_list = []
        def flatten_audios(obj):
            if isinstance(obj, (list, tuple)):
                if len(obj) > 0 and isinstance(obj[0], (float, int)):
                    audios_list.append(obj)
                else:
                    for item in obj: flatten_audios(item)
            elif isinstance(obj, (np.ndarray, torch.Tensor)):
                audios_list.append(obj)
        flatten_audios(audios)

        audios_np: List[np.ndarray] = []
        for a in audios_list:
            if isinstance(a, torch.Tensor): a = a.detach().cpu().numpy()
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            audios_np.append(a)

        input_features = None
        if audios_np:
            feat = self.feature_extractor(audios_np, sampling_rate=int(sampling_rate), return_tensors="np", return_attention_mask=False, padding="longest")
            input_features = feat["input_features"]
            if not isinstance(input_features, np.ndarray): input_features = np.asarray(input_features)

        tokenizer_kwargs = dict(tokenizer_kwargs or {})
        tokenizer_kwargs.setdefault("padding", "longest")
        tokenizer_kwargs.setdefault("add_special_tokens", False)
        tokenizer_kwargs["return_tensors"] = return_tensors

        enc = self.tokenizer(text, **tokenizer_kwargs)
        data: Dict[str, Any] = dict(enc)
        if input_features is not None:
            audio_tensor = torch.tensor(input_features, dtype=_resolve_torch_dtype(getattr(self, "audio_dtype", "float32")))
            data["input_features"] = audio_tensor
            data["audios"] = audio_tensor
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "input_features", "audios"]
