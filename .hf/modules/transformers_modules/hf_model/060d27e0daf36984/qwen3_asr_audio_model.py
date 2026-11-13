from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from .qwen3_asr_audio_config import Qwen3ASRAudioEncoderConfig


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    input_lengths = torch.clamp(input_lengths.long(), min=1)
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def _block_diagonal_attention_mask(hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    seq_length = int(hidden_states.shape[0])
    mask = torch.full(
        (1, 1, seq_length, seq_length),
        torch.finfo(hidden_states.dtype).min,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for i in range(1, int(cu_seqlens.numel())):
        start = int(cu_seqlens[i - 1].item())
        end = int(cu_seqlens[i].item())
        mask[..., start:end, start:end] = 0
    return mask


class Qwen3ASRAudioAttention(nn.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = int(config.d_model)
        self.num_heads = int(config.encoder_attention_heads)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("d_model must be divisible by encoder_attention_heads")
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del attention_mask
        seq_length = int(hidden_states.size(0))
        query = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim).transpose(0, 1)
        key = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim).transpose(0, 1)
        value = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim).transpose(0, 1)
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        mask = _block_diagonal_attention_mask(hidden_states, cu_seqlens)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq_length, self.embed_dim)
        return self.out_proj(attn_output)


class Qwen3ASRAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = int(config.d_model)
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.embed_dim, int(config.encoder_ffn_dim))
        self.fc2 = nn.Linear(int(config.encoder_ffn_dim), self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return (hidden_states,)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding requires an even channel count")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen3ASRAudioEncoder(PreTrainedModel):
    config_class = Qwen3ASRAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer"]

    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__(config)
        embed_dim = int(config.d_model)
        self.dropout = float(config.dropout)
        self.num_mel_bins = int(config.num_mel_bins)
        self.max_source_positions = int(config.max_source_positions)
        self.embed_scale = math.sqrt(embed_dim) if bool(config.scale_embedding) else 1.0
        self.n_window = int(config.n_window)
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([Qwen3ASRAudioEncoderLayer(config) for _ in range(int(config.encoder_layers))])
        self.ln_post = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, int(config.downsample_hidden_size), 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(int(config.downsample_hidden_size), int(config.downsample_hidden_size), 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(int(config.downsample_hidden_size), int(config.downsample_hidden_size), 3, 2, padding=1)
        conv_freq = ((((int(config.num_mel_bins) + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nn.Linear(int(config.downsample_hidden_size) * conv_freq, embed_dim, bias=False)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(embed_dim, int(config.output_dim))
        self.n_window_infer = int(config.n_window_infer)
        self.conv_chunksize = int(config.conv_chunksize)
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        return _block_diagonal_attention_mask(inputs_tensor, cu_seqlens)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        aftercnn_lens: Optional[torch.Tensor] = None,
    ):
        if feature_lens is None:
            feature_lens = torch.tensor([input_features.shape[-1]], dtype=torch.long, device=input_features.device)
        feature_lens = feature_lens.to(device=input_features.device, dtype=torch.long)
        if aftercnn_lens is None:
            aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        aftercnn_lens = aftercnn_lens.to(device=input_features.device, dtype=torch.long)

        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * int(chunk_num.sum().item()),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(int(length.item()), dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        bsz, channels, freq, time = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(bsz, time, channels * freq))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cnn_len_int = int(cnn_len.item())
            cu_chunk_lens += [window_aftercnn] * (cnn_len_int // window_aftercnn)
            remainder = cnn_len_int % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


__all__ = ["Qwen3ASRAudioEncoder", "Qwen3ASRAudioEncoderLayer", "Qwen3ASRAudioAttention"]
