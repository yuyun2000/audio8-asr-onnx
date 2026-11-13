from __future__ import annotations

from typing import Any, Dict, Optional, Union

from transformers import Qwen2Config, WhisperConfig

from .qwen3_asr_audio_config import Qwen3ASRAudioEncoderConfig


class ArkasrConfig(Qwen2Config):
    model_type = "arkasr"
    is_composition = True

    def __init__(
        self,
        whisper_config: Optional[Union[Dict[str, Any], WhisperConfig]] = None,
        qwen3_asr_audio_config: Optional[Union[Dict[str, Any], Qwen3ASRAudioEncoderConfig]] = None,
        adapter_type: str = "qwen3_asr_mlp_tower",
        merge_factor: int = 4,
        spec_aug: bool = False,
        use_rope: bool = False,
        max_whisper_length: int = 1500,
        mlp_adapter_act: str = "gelu",
        qwen3_asr_mlp_tower_layers: int = 4,
        qwen3_asr_mlp_tower_hidden_size: int = 0,
        qwen3_asr_mlp_tower_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(whisper_config, dict):
            self.whisper_config = WhisperConfig(**whisper_config)
        elif isinstance(whisper_config, WhisperConfig):
            self.whisper_config = whisper_config
        else:
            self.whisper_config = WhisperConfig()

        if isinstance(qwen3_asr_audio_config, dict):
            self.qwen3_asr_audio_config = Qwen3ASRAudioEncoderConfig(**qwen3_asr_audio_config)
        elif isinstance(qwen3_asr_audio_config, Qwen3ASRAudioEncoderConfig):
            self.qwen3_asr_audio_config = qwen3_asr_audio_config
        else:
            self.qwen3_asr_audio_config = Qwen3ASRAudioEncoderConfig(
                num_mel_bins=128,
                encoder_layers=18,
                encoder_attention_heads=14,
                encoder_ffn_dim=3584,
                d_model=896,
                max_source_positions=1500,
                n_window=50,
                output_dim=1024,
                n_window_infer=800,
                conv_chunksize=500,
                downsample_hidden_size=480,
            )

        self.adapter_type = adapter_type
        self.merge_factor = int(merge_factor)
        self.spec_aug = bool(spec_aug)
        self.use_rope = bool(use_rope)
        self.max_whisper_length = int(max_whisper_length)
        self.mlp_adapter_act = mlp_adapter_act
        self.qwen3_asr_mlp_tower_layers = int(qwen3_asr_mlp_tower_layers)
        self.qwen3_asr_mlp_tower_hidden_size = int(qwen3_asr_mlp_tower_hidden_size)
        self.qwen3_asr_mlp_tower_dropout = float(qwen3_asr_mlp_tower_dropout)

    def to_dict(self):
        output = super().to_dict()
        output["whisper_config"] = self.whisper_config.to_dict()
        output["qwen3_asr_audio_config"] = self.qwen3_asr_audio_config.to_dict()
        output["qwen3_asr_mlp_tower_layers"] = self.qwen3_asr_mlp_tower_layers
        output["qwen3_asr_mlp_tower_hidden_size"] = self.qwen3_asr_mlp_tower_hidden_size
        output["qwen3_asr_mlp_tower_dropout"] = self.qwen3_asr_mlp_tower_dropout
        return output


__all__ = ["ArkasrConfig"]
