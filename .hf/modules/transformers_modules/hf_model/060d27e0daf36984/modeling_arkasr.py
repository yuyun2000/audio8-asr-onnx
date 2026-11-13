from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2ForCausalLM
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_arkasr import ArkasrConfig
from .qwen3_asr_audio_config import Qwen3ASRAudioEncoderConfig
from .qwen3_asr_audio_model import Qwen3ASRAudioEncoder


class Qwen3AsrMlpTowerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_size = int(hidden_size)
        intermediate_size = int(intermediate_size or hidden_size * 4)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


class Qwen3AsrMlpTower(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)
        intermediate_size = int(intermediate_size or hidden_size * 4)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.layers = nn.ModuleList(
            [
                Qwen3AsrMlpTowerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_norm(hidden_states)


class ArkasrForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = ArkasrConfig
    base_model_prefix = "language_model"
    _no_split_modules = ["Qwen3ASRAudioEncoder"]
    _tied_weights_keys = ["language_model.lm_head.weight", "language_model.model.embed_tokens.weight"]

    def __init__(self, config: ArkasrConfig):
        super().__init__(config)
        self.audio_token_id = getattr(config, "audio_token_id", None)
        if self.audio_token_id is None:
            raise ValueError("`audio_token_id` must be defined in config.")

        audio_config = getattr(config, "qwen3_asr_audio_config", None)
        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        if audio_config is None:
            raise ValueError("`qwen3_asr_audio_config` must be defined in config.")

        self.language_model = Qwen2ForCausalLM(config)
        self.audio_encoder = Qwen3ASRAudioEncoder(audio_config)
        audio_dim = int(getattr(audio_config, "output_dim", 0) or 0)
        if audio_dim <= 0:
            raise ValueError("qwen3_asr_audio_config.output_dim must be positive.")

        layers = int(getattr(config, "qwen3_asr_mlp_tower_layers", 4) or 4)
        intermediate_size = int(getattr(config, "qwen3_asr_mlp_tower_hidden_size", 0) or audio_dim * 4)
        dropout = float(getattr(config, "qwen3_asr_mlp_tower_dropout", 0.0) or 0.0)
        self.audio_mlp_tower = Qwen3AsrMlpTower(
            hidden_size=audio_dim,
            num_layers=layers,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.audio_projector = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, int(config.hidden_size)),
        )
        self.all_tied_weights_keys = {}

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.language_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, *args, **kwargs):
        return self.language_model.resize_token_embeddings(*args, **kwargs)

    @staticmethod
    def _cache_seq_len(past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            try:
                return int(past_key_values.get_seq_length())
            except Exception:
                return 0
        try:
            return int(past_key_values[0][0].shape[-2])
        except Exception:
            return 0

    def _project_audio_row(
        self,
        input_features: torch.Tensor,
        token_count: int,
        dtype: torch.dtype,
        feature_length: Optional[int] = None,
    ) -> torch.Tensor:
        token_count = int(token_count)
        if token_count <= 0:
            return input_features.new_zeros((0, self.get_input_embeddings().embedding_dim), dtype=dtype)
        if input_features.dim() == 3 and input_features.size(0) == 1:
            input_features = input_features.squeeze(0)
        if input_features.dim() != 2:
            raise ValueError(f"Expected audio features with shape [mel, frames], got {tuple(input_features.shape)}")

        expected_mels = int(getattr(self.audio_encoder.config, "num_mel_bins", input_features.size(0)))
        if input_features.size(0) != expected_mels and input_features.size(1) == expected_mels:
            input_features = input_features.transpose(0, 1)
        if input_features.size(0) != expected_mels:
            raise ValueError(f"Audio feature bins mismatch: expected {expected_mels}, got {input_features.size(0)}")

        feature_length = int(feature_length) if feature_length is not None else int(input_features.size(1))
        feature_length = max(1, min(feature_length, int(input_features.size(1))))
        input_features = input_features[:, :feature_length]

        encoder_param = next(self.audio_encoder.parameters())
        encoded = self.audio_encoder(
            input_features.to(device=encoder_param.device, dtype=encoder_param.dtype),
            feature_lens=torch.tensor([feature_length], dtype=torch.long, device=encoder_param.device),
        )
        hidden = getattr(encoded, "last_hidden_state", encoded)
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        if hidden.dim() == 3 and hidden.size(0) == 1:
            hidden = hidden.squeeze(0)
        if hidden.dim() != 2:
            raise ValueError(f"Expected audio encoder output [time, dim], got {tuple(hidden.shape)}")

        tower_param = next(self.audio_mlp_tower.parameters())
        hidden = self.audio_mlp_tower(hidden.to(device=tower_param.device, dtype=tower_param.dtype))
        projector_param = next(self.audio_projector.parameters())
        hidden = hidden.to(device=projector_param.device, dtype=projector_param.dtype)
        if int(hidden.size(0)) != token_count:
            hidden = F.adaptive_avg_pool1d(
                hidden.transpose(0, 1).float().unsqueeze(0),
                output_size=token_count,
            ).squeeze(0).transpose(0, 1).to(dtype=projector_param.dtype)
        return self.audio_projector(hidden).to(dtype=dtype)

    def _inject_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        feature_lens: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeddings = self.get_input_embeddings()(input_ids).clone()
        if input_features is None:
            input_features = audios
        if feature_lens is None:
            feature_lens = audio_feature_lengths
        if input_features is None:
            return input_embeddings

        input_features = input_features.to(device=input_embeddings.device)
        if input_features.dim() == 4 and input_features.size(1) == 1:
            input_features = input_features.squeeze(1)
        if feature_lens is not None and not torch.is_tensor(feature_lens):
            feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=input_embeddings.device)
        if torch.is_tensor(feature_lens):
            feature_lens = feature_lens.to(device=input_embeddings.device)

        audio_mask = input_ids.eq(self.audio_token_id)
        for batch_i in range(int(input_ids.size(0))):
            positions = torch.nonzero(audio_mask[batch_i], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            feature_length = None
            if torch.is_tensor(feature_lens) and batch_i < int(feature_lens.numel()):
                feature_length = int(feature_lens[batch_i].item())
            projected = self._project_audio_row(
                input_features[batch_i],
                int(positions.numel()),
                input_embeddings.dtype,
                feature_length=feature_length,
            )
            input_embeddings[batch_i, positions, :] = projected.to(device=input_embeddings.device)
        return input_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        feature_lens: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        past_len = self._cache_seq_len(past_key_values)
        audio_inputs = input_features if input_features is not None else audios
        audio_lengths = feature_lens if feature_lens is not None else audio_feature_lengths
        if audio_inputs is not None and input_ids is not None and past_len == 0:
            inputs_embeds = self._inject_audio_embeddings(
                input_ids=input_ids,
                input_features=audio_inputs,
                feature_lens=audio_lengths,
            )

        outputs = self.language_model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs[0]

        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_for_logits = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, torch.Tensor):
            hidden_for_logits = hidden_states[:, logits_to_keep, :]
        else:
            hidden_for_logits = hidden_states
        logits = self.language_model.lm_head(hidden_for_logits)

        loss = None
        if labels is not None:
            loss = self.language_model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        past_len = self._cache_seq_len(past_key_values)
        if past_len > 0:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "input_features": kwargs.get("input_features", kwargs.get("audios", None)),
            "feature_lens": kwargs.get("feature_lens", kwargs.get("audio_feature_lengths", None)),
        }
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            del model_inputs["input_ids"]
        return model_inputs


__all__ = [
    "ArkasrForConditionalGeneration",
    "Qwen3AsrMlpTower",
    "Qwen3AsrMlpTowerBlock",
]
