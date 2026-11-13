# Audio8-ASR-0.1B (`arkasr`) — authoritative inference spec for a sherpa-onnx C++ port

Scope: `Edge0/Audio8-ASR-0.1B` (transformers checkpoint) and `Edge0/Audio8-ASR-0.1B-onnx-runtime`
(official fp32 ONNX bundle). Everything below is either read out of the released code/graphs or
measured; each claim carries `file:line` evidence and, where it matters, a measurement.

All paths are relative to `/Users/yunyu/Documents/DSH/AX-model/audio8/`.

* `downloaded/hf_model/` — HF checkpoint + remote code (28 files, 646 MB incl. `model.safetensors`)
* `downloaded/onnx_release/` — official ONNX runtime release (fp32 graphs only, 2.5 GB)
* `downloaded/wavs/` — test audio (16 kHz mono)
* `ref/` — runners, dumps, parity JSON
* `reports/` — this document

---

## 0. TL;DR

| item | value |
|---|---|
| architecture | `ArkasrForConditionalGeneration` = Qwen2 decoder (8L/512d) + `Qwen3ASRAudioEncoder` (18L/896d) + 4-block MLP tower (1024) + LayerNorm+Linear projector (1024→512) |
| audio front end | Whisper-style log-mel, **128 bins**, `n_fft=400`, `hop=160`, 30 s cap, pad to **3000 frames**, log10 + `(x+4)/4` |
| audio token rule | **prompt placeholders**: `max(1, floor(floor(L/160)+1)/2/4)` — floor semantics |
| encoder frames | `(T//100)*13 + ceil((T%100)/8)`, i.e. **13 tokens per 100-frame (1 s) chunk**, 8× time downsample |
| max @30 s | encoder produces **390** valid rows, prompt gets **375** `<|audio|>` tokens (pooled) |
| prompt | `<|user|><|begin_of_audio|>` + `<|audio|>`×N + `<|end_of_audio|>` + `Please transcribe this audio.` + `<|assistant|>` (**no system prompt**) |
| RoPE | `use_rope:false` in `config.json` is **not honored** — RoPE (θ=1e6) *is* applied in every LM graph |
| KV cache | 8 layers × K/V × `[1,8,512,64]` fp32 = **16.78 MB**, `max_total_len = 512` |
| vocab | 151936 embedding rows / 151652 tokenizer entries; `tie_word_embeddings=true` |
| EOS | stop on 151645 `<|im_end|>` or 151643 `<|endoftext|>`; block {151646,151647,151648,151649,151650} |
| measured | bundle vs transformers: **5/5 transcripts byte-identical**; audio hidden cosine **1.000000**, prefill logits max_abs **3.4e-5** |

---

## 1. Files and sizes (task 3)

`model_bundle/` of `Edge0/Audio8-ASR-0.1B-onnx-runtime` (byte-exact sizes from the HF API, confirmed on disk):

| file | bytes | used | notes |
|---|---:|---|---|
| `audio_hidden.onnx` | 880,238,173 | **yes** | 1108 nodes, 327 initializers = 879.8 MB |
| `lm_cache_prefill.onnx` | 414,144,056 | **yes** | 1318 nodes, 98 initializers = 414.0 MB |
| `lm_cache_decode.onnx` | 414,145,009 | **yes** | 943 nodes, 98 initializers = 414.0 MB |
| `lm_logits.onnx` | 414,132,615 | fallback | 1209 nodes, 98 initializers = 414.0 MB |
| `weights/token_embedding.npy` | 311,165,056 | **yes** | `(151936, 512)` float32, C-order |
| `weights/audio_projector.npz` | 2,108,430 | **yes** | `norm_weight (1024,)`, `norm_bias (1024,)`, `linear_weight (512,1024)`, `linear_bias (512,)` |
| `tokenizer.json` | 11,419,524 | **yes** | fast tokenizer, 151652 entries |
| `vocab.json` | 2,776,833 | optional | BPE vocab |
| `merges.txt` | 1,671,853 | optional | BPE merges |
| `added_tokens.json` | 238 | optional | 151643…151651 map |
| `tokenizer_config.json` / `special_tokens_map.json` | 2,185 / 434 | optional | |
| `preprocessor_config.json` | 355 | **stale, do not use** | says `feature_size: 80` (see §11.2) |
| `qwen3_asr_feature_extractor/preprocessor_config.json` | 357 | **yes** | `feature_size: 128` |
| `metadata.json` | 2,806 | **yes** | |
| `config.json` | 3,591 | reference | |
| `processor_config.json` | 123 | reference | `merge_factor: 4`, `audio_dtype: bfloat16` |
| `chat_template.jinja` | 327 | **unused** | see §11.5 |
| `audio8_audio_wrapper_config.json` | 375 | reference | |
| `generation_config.json` | 147 | reference | |

Quantization variants exist and were **not** downloaded (we use fp32 per instructions):
`audio_hidden_int8.onnx` 234,699,007 · `lm_cache_{prefill,decode}_int8.onnx` 1.31/1.26 MB + `.onnx.data` 103.6/103.6 MB ·
`lm_cache_{prefill,decode}_int4.onnx` 165/134 kB + `.onnx.data` 55.5/55.5 MB.
The README says the *default* runtime path is int8/int8; the fp32 path is the reference path.

HF checkpoint (`Edge0/Audio8-ASR-0.1B`, used for the transformers reference): `model.safetensors` 648,031,000 B,
429 tensors, **all BF16**, 323,990,528 parameters; `tokenizer.json` 11,419,524; `merges.txt`; `vocab.json`;
`*.py` (5 modules); `chat_template.jinja`; `preprocessor_config.json` (`feature_size: 128`).

---

## 2. Log-mel featurization

### 2.1 Parameters

`downloaded/onnx_release/model_bundle/qwen3_asr_feature_extractor/preprocessor_config.json`:

```
feature_extractor_type: WhisperFeatureExtractor   feature_size: 128      n_fft: 400
hop_length: 160      chunk_length: 30      n_samples: 480000      nb_max_frames: 3000
sampling_rate: 16000   dither: 0.0   padding_side: right   padding_value: 0.0
```

The runtime deliberately prefers the sub-directory over the stale top-level file:
`asr_onnx_runtime.py:202-205` (`feature_dir = bundle/"qwen3_asr_feature_extractor"; if not exists: bundle`).
The HF checkpoint's own `preprocessor_config.json:8` also says `feature_size: 128`.

### 2.2 Exact computation (`transformers/models/whisper/feature_extraction_whisper.py`)

* filterbank: `mel_filter_bank(num_frequency_bins=1+n_fft//2=201, num_mel_filters=128, min_frequency=0.0,
  max_frequency=8000.0, sampling_rate=16000, norm="slaney", mel_scale="slaney")` — lines 95–103.
* `_torch_extract_fbank_features` lines 141–167:
  1. `window = torch.hann_window(400)` (periodic),
  2. `stft = torch.stft(w, 400, 160, window, return_complex=True)` → default `center=True`, `pad_mode="reflect"`, onesided,
  3. `magnitudes = stft[..., :-1].abs()**2` — **the last STFT frame is dropped**, so `mel_frames = floor(L/160)`,
  4. `mel_spec = mel_filters.T @ magnitudes` → `[128, floor(L/160)]`,
  5. `log_spec = clamp(mel_spec, min=1e-10).log10()`,
  6. `log_spec = max(log_spec, global_max - 8.0)` (single-utterance path, line 161),
  7. `log_spec = (log_spec + 4.0) / 4.0`.
* dtype fed to the graph: **float32**. (The HF *processor* casts features to `audio_dtype` from
  `processor_config.json` = **bfloat16** — a lossy step the ONNX path does not have; §11.6.)

### 2.3 Padding / slicing / 30 s cap (`asr_onnx_runtime.py:244-266`)

```
max_samples = 30 * 16000                                   # 480000            (l.245)
audio = audio[:max_samples]                                # hard truncation   (l.246-247)
sample_count = max(1, len(audio))                                             (l.248)
feature = fe([audio], padding="longest", max_length=480000, ...)  # [1,128,F]  (l.249-256)
hop_length = fe.hop_length                                 # 160               (l.257)
encoder_feature_len = ceil(sample_count / 160)             # NOTE: ceil         (l.258)
encoder_feature_len = min(max(1, that), feature.shape[-1])                     (l.259)
frames_padded = 3000  (metadata.graphs.audio_hidden.frames_padded)            (l.261)
feature = pad(feature, ((0,0),(0,0),(0, 3000-F))) or feature[..., :3000]       (l.262-265)
```

Because `padding="longest"` is used with a single utterance, `WhisperFeatureExtractor.pad()` does **not** pad —
`max_length` only drives truncation (`feature_extraction_whisper.py:302-311`). The 3000-frame pad is therefore done
explicitly in numpy, right-aligned zeros.

Measured (`ref/tensors/*.npz`): `mel.shape == (1, 128, 3000)` for every test wav; `ref/run_bundle.py` prints it.

**C++ requirement.** Produce `[1,128,3000]` float32: `floor(L/160)` frames of real features (max 3000) followed by
zeros. Frame `t` covers samples `[160t-200, 160t+200)` (centred, reflect-padded), so the reproduce path is:
hann(400, periodic) → rFFT(400) → |·|² dropping the last frame → 201×128 slaney mel → log10(clamp 1e-10) →
`max(x, max-8)` → `(x+4)/4`. Do **not** use `povey`/HTK mel or per-utterance mean/variance normalisation; there is none.

---

## 3. Prompt token ids fed at prefill

### 3.1 Text template

`asr_onnx_runtime.py:232-242` (`_build_prompt`):

```python
audio_tokens = "<|audio|>" * N
prompt = "<|user|>" + "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>" \
         + "Please transcribe this audio." + "<|assistant|>" + response_prefix
```

`response_prefix` is `""` (`metadata.json:7`), and `language` is ignored (`_build_prompt` drops it, l.233;
`metadata.json:8` `language_hint_enabled: false`).

This is byte-identical to the HF processor's own builder: `processing_arkasr.py:187-189` (role → token),
`:243` (`bos + MARKER + eos`), `:365-366` (marker → N copies), `:252-255` (`add_generation_prompt` appends
`<|assistant|>` when the last role is `user`).

**There is no system prompt.** `chat_template.jinja` (which would emit
`<|im_start|>system\nYou are a helpful assistant<|im_end|>\n`) is dead code — `ArkasrProcessor.apply_chat_template`
takes `chat_template` and explicitly ignores it (`processing_arkasr.py:296-297`), building the string by hand instead.

### 3.2 Ids

`metadata.json:17-38` + `tokenizer_config.json` + `added_tokens.json`:

| token | id |
|---|---:|
| `<|endoftext|>` (pad) | 151643 |
| `<|im_start|>` | 151644 |
| `<|im_end|>` (eos) | 151645 |
| `<|audio|>` (placeholder, **not** flagged special) | **151646** |
| `<|user|>` | 151647 |
| `<|begin_of_audio|>` | 151648 |
| `<|end_of_audio|>` | 151649 |
| `<|assistant|>` | 151650 |
| `<|system|>` | 151651 |

`"Please transcribe this audio."` under Qwen2-BPE (`add_special_tokens=False`) = 6 tokens:
`[5501, 1356, 3114, 419, 7699, 13]` = `Please · Ġtrans · cribe · Ġthis · Ġaudio · .`

### 3.3 Full sequence (measured, `ref/tensors/en.npz` → `prompt_ids`, 99 tokens for a 7.152 s wav)

```
[0]      151647  <|user|>
[1]      151648  <|begin_of_audio|>
[2..90]  151646  × 89        <|audio|>            (N = 89 audio tokens)
[91]     151649  <|end_of_audio|>
[92..97] 5501 1356 3114 419 7699 13   "Please transcribe this audio."
[98]     151650  <|assistant|>
```

So `prompt_len = N_audio + 10`, and `N_audio` = number of audio placeholder ids (the runtime locates them by
scanning `input_ids == 151646`, `asr_onnx_runtime.py:304`).

Measured on all 5 wavs (`ref/transcripts.json`): `N_audio / prompt_len` = 89/99, 70/80, 125/135, 220/230, 209/219.

---

## 4. How the `<|audio|>` token is expanded to N tokens

### 4.1 The prompt-side rule (floor semantics)

`processing_arkasr.py:162-165`:

```python
def calculate_audio_token_count(self, mel_frames):
    downsampled = (mel_frames + 1) // 2
    merged = downsampled // self.merge_factor        # merge_factor = 4
    return max(int(merged), 1)
```

with `mel_frames = effective_len // hop_length` (`:281`, floor) and `effective_len = min(len(audio), audio_max_length)`.
The ONNX runtime reimplements exactly this as `ark_audio_token_count` (`asr_onnx_runtime.py:113-117`):

```
N_audio = max(1, floor( floor(L/160) + 1 ) / 2 ) / 4 )        # L = min(#samples, 480000)
```

`metadata.json:13` names this rule `arkasr_processor_floor_mel_downsample_merge` — it matches the code.

### 4.2 The encoder-side rule (chunk of 100 frames → 13 tokens)

`qwen3_asr_audio_model.py:17-21`:

```python
def _get_feat_extract_output_lengths(L):
    leave = L % 100
    feat  = (leave - 1) // 2 + 1                      # ceil(leave/2)   = conv1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (L // 100) * 13
```

which collapses to

```
E_valid = (L // 100) * 13 + ceil( (L % 100) / 8 )
```

because three successive `ceil(·/2)` stages compose to `ceil(·/8)`. **13 tokens per 100-frame chunk**
(= 1 s of audio; 100 → 50 → 25 → 13 through the three strided convs, each `ceil(in/2)`),
and `13 * 30 = 390` at 3000 frames — exactly `metadata.graphs.audio_hidden.max_audio_tokens = 390` and
`valid_mask_sum_at_max = 390` (`metadata.json:50-53`).

### 4.3 The two numbers differ, and that is by design

`N_audio` (prompt placeholders) ≤ `E_valid` (encoder rows). The bridge is **adaptive average pooling**
(`modeling_arkasr.py:184-188`, `asr_onnx_runtime.py:288-291`):

```python
valid_hidden = hidden[valid_mask]                    # [E_valid, 1024]
if valid_hidden.shape[0] != N_audio:
    valid_hidden = adaptive_avg_pool1d(...)          # → [N_audio, 1024]
```

Measurements (`ref/transcripts.json`, `ref/parity.json`):

| wav | samples | mel frames | `E_valid` (valid_mask sum) | `N_audio` (prompt) |
|---|---:|---:|---:|---:|
| en.wav | 114,432 | 715 | **93** | **89** |
| zh.wav | 89,472 | 559 | **73** | **70** |
| 0.wav | 160,850 | 1,005 | **131** | **125** |
| 4.wav | 282,240 | 1,764 | **229** | **220** |
| en_zipformer1.wav | 267,440 | 1,671 | **217** | **209** |
| synthetic 30 s | 480,000 | 3,000 | **390** | **375** |

`adaptive_avg_pool1d` semantics (PyTorch): `start = floor(i*E/N)`, `end = ceil((i+1)*E/N)`;
the runtime's numpy twin is `adaptive_avg_pool_time` (`asr_onnx_runtime.py:127-138`), same formula.

---

## 5. Audio encoder math

Source: `downloaded/hf_model/qwen3_asr_audio_model.py`, config `qwen3_asr_audio_config` in `config.json:200-280`.

### 5.1 Hyper-parameters

`num_mel_bins 128 · encoder_layers 18 · encoder_attention_heads 14 (head_dim 64) · d_model 896 ·
encoder_ffn_dim 3584 · activation gelu (exact erf) · max_source_positions 1500 · n_window 50 ·
n_window_infer 800 · conv_chunksize 500 · downsample_hidden_size 480 · output_dim 1024 · scale_embedding false`

### 5.2 Front end (conv) — `qwen3_asr_audio_model.py:130-152, 175-202`

```
conv2d1: Conv2d(1   → 480, k=3, s=2, p=1) + gelu      # (l.142, 195)
conv2d2: Conv2d(480 → 480, k=3, s=2, p=1) + gelu      # (l.143, 196)
conv2d3: Conv2d(480 → 480, k=3, s=2, p=1) + gelu      # (l.144, 197)
conv_freq  = ceil(ceil(ceil(128/2)/2)/2) = 16         # (l.145)
conv_out: Linear(480*16 = 7680 → 896, bias=False)     # (l.146, 201)
```

Time: 100 → 50 → 25 → 13 per chunk (×0.125). Frequency: 128 → 64 → 32 → 16.
The full sequence is **split into 100-frame chunks first and convolution runs batched over chunks**
(`:175-186`), i.e. convolution never crosses a chunk boundary.

### 5.3 Positional information — `:107-122, 203-208`

A fixed sinusoidal table of shape `[1500, 896]`:
`log_inc = ln(10000)/(448-1)`, `inv = exp(-log_inc*arange(448))`,
`table = cat([sin(t*inv), cos(t*inv)])`.

Critically it is indexed **per chunk**: `table[:13]` is added to every chunk's 13 positions
(`positional_embedding[:padded_embed.shape[1]]` with `padded_embed` = `[n_chunks, 13, 896]`). So all chunks share
positions 0…12. In the ONNX graph this appears as a single `Constant /Constant_19_output_0` of shape `(1,13,896)`
added by `Add('/conv_out/MatMul_output_0', '/Constant_19_output_0')`; verified equal to `table[:13]` to 7e-7.

### 5.4 Encoder transformer — `:210-226`

```
window_aftercnn = 13 * (n_window_infer // (n_window*2)) = 13 * 8 = 104     (l.211)
cu_seqlens: split the valid token stream into blocks of 104 (last block = remainder)  (l.212-218)
per layer: pre-LN → MHA (14 heads, scale 64**-0.5, block-diagonal additive mask) → residual
           → pre-LN → fc1(896→3584) → gelu → fc2(3584→896) → residual              (l.90-104)
head: ln_post(896) → proj1(896→896) → gelu → proj2(896→1024)                       (l.223-226)
```

So encoder self-attention is **windowed to 104 tokens = 8 chunks = 800 frames = 8 s**; there is no causal mask and
no cross-window attention. 390 valid tokens at 30 s = blocks `[104, 104, 104, 78]` (measured on the graph: mask
`Where_2` allowed 38532 = 3·104² + 78² pairs — exactly block-diagonal with those sizes).

### 5.5 MLP tower + projector — `modeling_arkasr.py:40-71, 107-110`

```
audio_mlp_tower: 4 × [ x + fc2(gelu(fc1(LN(x)))) ], 1024 → 4096 → 1024, then final_norm(LN 1024)
audio_projector: LayerNorm(1024, eps 1e-5) → Linear(1024 → 512)
```

The tower is **inside** `audio_hidden.onnx` (its output `audio_hidden` is already 1024-d and post-`final_norm`);
only the projector lives outside, as `weights/audio_projector.npz`, and is applied **after** pooling
(`asr_onnx_runtime.py:292-298`; `modeling_arkasr.py:184-189`).

### 5.6 Order of operations (authoritative)

```
mel[1,128,3000] ──► chunk [30,1,128,100] ──► conv×3+gelu ──► [30,480,16,13]
   ──► conv_out ──► [30,13,896] ──► +sin_table[:13] ──► flatten [390,896]
   ──► 18× encoder (window 104) ──► ln_post/proj1/gelu/proj2 ──► [390,1024]
   ──► MLP tower ×4 + final_norm ──► audio_hidden[390,1024]   (+ audio_valid_mask[390])
   ──► keep rows where mask ──► [E_valid,1024] ──► adaptive-avg-pool to N_audio
   ──► LayerNorm+Linear (projector) ──► [N_audio,512]  == audio embeddings
```

---

## 6. Language model

`modeling_arkasr.py:92` — `self.language_model = Qwen2ForCausalLM(config)` where `config` is `ArkasrConfig(Qwen2Config)`
(`configuration_arkasr.py:10`). Values from `config.json`:

```
hidden_size 512 · num_hidden_layers 8 · num_attention_heads 8 (head_dim 64) · num_key_value_heads 8 (MHA)
intermediate_size 1408 · hidden_act silu · rms_norm_eps 1e-6 · tie_word_embeddings true
rope_theta 1e6 · max_position_embeddings 32768 · layer_types: 8× full_attention · use_sliding_window false
use_rope false  ← NOT honored, see 6.2
```

Per-tensor facts from the checkpoint header (429 tensors, all BF16): `q_proj/k_proj/v_proj` have **bias**,
`o_proj` and `mlp.{gate,up,down}_proj` do **not**; `input_layernorm`/`post_attention_layernorm`/`model.norm` are
weight-only RMSNorm; **there is no `lm_head.weight` tensor** — it is tied to `embed_tokens`
(`_tied_weights_keys`, `modeling_arkasr.py:78`; `config.json:44`).

### 6.1 MLP

SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`, 512 → 1408 → 512.

### 6.2 Positional encoding — RoPE **is** used (contradicts `use_rope:false`)

* `use_rope: false` (`config.json:47`) is only stored by `ArkasrConfig` (`configuration_arkasr.py:21, 60`) and is
  **never read** by any module — `modeling_arkasr.py` never references `config.use_rope`, and `Qwen2ForCausalLM`
  always builds `rotary_emb` with `rope_theta = 1e6`.
* Every LM graph contains exactly one `Cos` and one `Sin` node (`ref/onnx_io.json` → `has_sin_cos: true`).
* Probe (`ref/parity.json` → `rope_probe_*`, run through `lm_cache_prefill.onnx`):
  - shifting **all** `cache_position` by +64 leaves the logits unchanged to 1.2–1.6e-5 (pure fp noise, top-1 never
    changes). That is the signature of **relative** rotation (a global rotation of every position cancels in q·k).
  - forcing **all** positions to 0 changes the logits a lot (cosine 0.55–0.93, max_abs 11.6–15.1).
  ⇒ positional information is present, relative, and comes from RoPE.

So: a C++ port must implement RoPE with `theta = 1e6`, `head_dim = 64`, standard rotate-half, interleaved by pair
index (`q[2i], q[2i+1]`), applied to q and k only.

### 6.3 Graph split used by the official bundle (`metadata.json:67-93`, `asr_onnx_runtime.py:441-687`)

| stage | graph | contract |
|---|---|---|
| prefill | `lm_cache_prefill.onnx` | in `inputs_embeds [1,S,512] f32`, `cache_position [S] int64`; out `logits [1,S,151936]` + 16 cache deltas `key/value_delta_i [1,8,S,64]` |
| decode | `lm_cache_decode.onnx` | in `inputs_embeds [1,1,512]`, `attention_mask [1,512] int64`, `cache_position [1] int64`, `cache_key/value_i [1,8,512,64]`; out `logits [1,1,151936]` + 16 deltas `[1,8,1,64]` |
| no-cache fallback | `lm_logits.onnx` | in `inputs_embeds [1,S,512]`, `attention_mask [1,S]`; out `logits [1,151936]` (2-D, **last position only**) |

`lm_cache_prefill` has **no** `attention_mask` input: prefill is always a full causal pass over positions
`cache_position[i]` (caller passes `arange(S)`); the mask is implicitly causal.
`lm_logits.onnx` has no `cache_position` input at all — positions are taken as `0..S-1` internally, so it is only
valid as a single full-context pass (which is why the runtime always feeds it the whole sequence,
`asr_onnx_runtime.py:394-404`).

The runtime's prefill takes **only the last row** of the prefill logits (`outputs[0][0, -1, :]`,
`asr_onnx_runtime.py:559`) and then one decode step per generated token.

### 6.4 KV cache

```
shape per tensor: [1, num_kv_heads=8, max_total_len=512, head_dim=64] float32
count:            2 × 8 layers = 16 tensors  → 16 × 1,048,576 B = 16.78 MB
allocation:       asr_onnx_runtime.py:531-545 (zeros), dtype from the decode graph's `cache_key_0` input
prefill write:    caches[2i][:, :, :S, :] = delta_i         (l.560-564)
decode write:     caches[2i][:, :, pos:pos+1, :] = delta_i   (l.594-598)
```

Cache layout is `[batch, kv_head, position, head_dim]` — i.e. **sequence-major last**, unlike the usual
`[B,H,T,D]` HF layout it is exactly the same, but note the graph expects the *full padded* 512-length buffers on
every decode call (not a growing slice) plus an attention mask.

### 6.5 Decode loop (`asr_onnx_runtime.py:644-667`)

```
logits = prefill(embeds_prompt)                     # [151936]
for step in range(max_new_tokens):
    logits = mask/penalise/boost(logits)
    next = argmax(logits)
    if next in {151645} or next == 151643: stop
    append next; feed embed(next) at position=current_position,
        attention_mask[0, :current_position+1] = 1, then current_position += 1
```

`attention_mask` is rebuilt as zeros(512) with a ones-prefix of length `valid_len = position+1`
(`asr_onnx_runtime.py:580-581`); `cache_position = position` (`l.585`).

### 6.6 Logit post-processing (`asr_onnx_runtime.py:312-317, 405-413`)

* `repetition_penalty = 1.0` by default → no-op.
* `_mask_logits`: `logits[151646] = logits[151647] = logits[151648] = logits[151649] = logits[151650] = -inf`
  (`extra_block_token_ids`, `metadata.json:30-36`); `asr_block_token_id_from = -1` → no suffix block.
* greedy (`do_sample=false`, `metadata.json:99-104`); temperature/repetition defaults are inert.

---

## 7. Tokenizer and detokenisation

* Class: `Qwen2Tokenizer` (`tokenizer_config.json:96`), BPE, vocab 151652, byte-level (`Ġ` = space, `Ċ` = newline).
  The ONNX runtime loads the raw fast tokenizer (`tokenizers.Tokenizer.from_file(bundle/"tokenizer.json")`,
  `asr_onnx_runtime.py:201`) and encodes the prompt with `add_special_tokens=False` (`l.302`).
* Decode in the runtime: `tokenizer.decode(generated, skip_special_tokens=False)` (`l.422`, `l.669`) followed by
  `normalize_prediction_text` (`l.87-98`):
  1. truncate at the first of `<|user|>`, `<|assistant|>`, `<|im_end|>` (`l.30, 78-84`);
  2. cut everything before `<|text|>` / `<asr_text>` if present;
  3. drop a leading `language XX` prefix;
  4. strip all `<|...|>` tokens (`SPECIAL_TOKEN_PATTERN`, `l.23-29`);
  5. collapse whitespace, strip, then strip leading `[\s,.;:!?-]+` (`l.31`).
* **`<|audio|>` (151646) is not marked special** in `tokenizer_config.json` (`"special": false`), so
  `skip_special_tokens=True` would *keep* it. A C++ decoder must either rely on the -inf blocking (§6.6) or filter
  it explicitly.
* Role/eos tokens do decode verbatim with `skip_special_tokens=False` — e.g.
  `[151647,151648,151646,151649,5501,…,13,151650]` → `'<|user|><|begin_of_audio|><|audio|><|end_of_audio|>Please transcribe this audio.<|assistant|>'`.
* Measured raw output for `en.wav`: `'The tribal chief then called for the boy and presented him with fifty pieces
  of gold.<|im_end|>'` → normalised to the final transcript.

---

## 8. ONNX graph IO table (task 3)

From `ref/inspect_graphs.py` → `ref/onnx_io.json` (opset `ai.onnx:18`, producer `pytorch 2.8.0`).

### `audio_hidden.onnx` — 880.2 MB, 1108 nodes, 327 initializers (879.8 MB)

| dir | name | dtype | shape |
|---|---|---|---|
| in | `audios` | float32 | `[1, 128, 3000]` |
| in | `audio_feature_lengths` | int64 | `[1]` |
| out | `audio_hidden` | float32 | `[390, 1024]` |
| out | `audio_valid_mask` | int64 | `[390]` |

ops: `Constant×361, Gemm×118, Transpose×93, Add×89, Mul×89, Reshape×75, Unsqueeze×61, LayerNormalization×42,
MatMul×37, Div×31, Erf×26, Slice×19, Softmax×18, Cast×15, Sub×6, And×4, Gather×3, Clip×3, Conv×3, Where×3`
(`Conv`: k=3, s=2, pads=[1,1,1,1] ×3; `Gemm` transB=1; GELU = `Div(√2)→Erf→+1→Mul→×0.5`).
The whole encoder is statically shaped to 30 chunks: `Reshape(audios[0] → [30,100,128])` with
`Constant_7 = [30,100,128]`, `Constant_20 = [390,-1]`, `Constant_42 = 104` (attention window).

### `lm_cache_prefill.onnx` — 414.1 MB, 1318 nodes, 98 initializers (414.0 MB)

| dir | name | dtype | shape |
|---|---|---|---|
| in | `inputs_embeds` | float32 | `[1, 'sequence', 512]` |
| in | `cache_position` | int64 | `['sequence']` |
| out | `logits` | float32 | `[1, 'sequence', 151936]` |
| out | `key_delta_0..7` / `value_delta_0..7` | float32 | `[1, 8, 'sequence', 64]` |

### `lm_cache_decode.onnx` — 414.1 MB, 943 nodes, 98 initializers (414.0 MB)

| dir | name | dtype | shape |
|---|---|---|---|
| in | `inputs_embeds` | float32 | `[1, 1, 512]` |
| in | `attention_mask` | int64 | `[1, 512]` |
| in | `cache_position` | int64 | `[1]` |
| in | `cache_key_0..7`, `cache_value_0..7` | float32 | `[1, 8, 512, 64]` |
| out | `logits` | float32 | `[1, 1, 151936]` |
| out | `key_delta_0..7` / `value_delta_0..7` | float32 | `[1, 8, 1, 64]` |

### `lm_logits.onnx` — 414.1 MB, 1209 nodes, 98 initializers (414.0 MB)

| dir | name | dtype | shape |
|---|---|---|---|
| in | `inputs_embeds` | float32 | `[1, 'sequence', 512]` |
| in | `attention_mask` | int64 | `[1, 'sequence']` |
| out | `logits` | float32 | `['Castlogits_dim_0', 151936]` → `[1, 151936]` |

All three LM graphs carry ~103.5 M parameters: the 25.7 M Qwen2 stack **plus a materialised 77.8 M `lm_head`
matrix** (the tied embedding), which is why each is 414 MB even though embeddings are an *input*.
Parameter budget of the bundle: audio 219.9 M + LM 103.5 M + separate embedding table 77.8 M + projector 0.53 M.

---

## 9. Measured runs (tasks 4 and 5)

Engine: official `OnnxCacheAsrEngine` (fp32/fp32), ORT 1.23.2, CPU, greedy, `max_new_tokens=128`.
Reference: `AutoProcessor` + `AutoModelForCausalLM` (`trust_remote_code=True`, `torch_dtype=float32`,
`attn_implementation="eager"`, greedy, `max_new_tokens=128`, `audio_max_length=30*16000`).

| wav | language | audio s | bundle text | transformers text | match | bundle s | RTF |
|---|---|---:|---|---|---|---:|---:|
| `en.wav` | en | 7.15 | The tribal chief then called for the boy and presented him with fifty pieces of gold. | same | ✅ | 1.83 | 0.257 |
| `zh.wav` | zh | 5.59 | 开放时间早上九点至下午五点。 | same | ✅ | 1.37 | 0.246 |
| `0.wav` | en+zh | 10.05 | 昨天是 Monday。Today is礼拜二。The day after tomorrow，是星期三。 | same | ✅ | 1.28 | 0.127 |
| `4.wav` | zh (+en) | 17.64 | 嗯，On time 就准时，in time 是及时交。他总是准时交他的作业。那用一般现在时是没有什么感情色彩的陈述一个事实。下一句话为什么要用现在进行时？他的意思并不是说说他现在正在教他的。 | same | ✅ | 1.59 | 0.090 |
| `en_zipformer1.wav` | en | 16.72 | God, as a direct consequence of the sin which man thus punished, had given her a lovely child, whose place was on that same dishonored bosom to connect her parent for ever with the race and descent of mortals, and to be finally a blessed soul in heaven. | same | ✅ | 1.43 | 0.086 |

All five stop naturally on `151645` (`hit_stop: true`). `ref/transcripts.json` holds the machine-readable form
(`bundle_text`, `transformers_text`, `timing`, `mel_shape`, `valid_mask_sum`, `prompt_len`, `hit_stop`,
`stop_token_id`, RTF, engine load time 1.04 s).
Tensors for all five wavs are in `ref/tensors/*.npz`: `mel [1,128,3000] f32`, `prompt_ids [P] i64`,
`inputs_embeds [P,512] f32`, `audio_embeddings [N,512] f32`, `audio_hidden [390,1024] f32`,
`valid_mask [390]`, `audio [samples] f32`; the transformers side is in `ref/tensors_hf/*.npz`.

### 9.1 Numerical parity (task: prove the spec = the graphs) — `ref/parity.json`

| wav | mel frames | valid rows | `audio_hidden` cosine / max_abs | projector cosine | prefill logits cosine / max_abs | `lm_logits` vs prefill |
|---|---:|---:|---|---|---|---:|
| en | 715 | 93 | 1.000000 / 1.0e-4 | 1.000000 | 1.000000 / 3.4e-5 | 0.0 (bit-exact) |
| zh | 559 | 73 | 1.000000 / 1.7e-4 | 1.000000 | 1.000000 / 2.6e-5 | 0.0 |
| 0 | 1005 | 131 | 1.000000 / 6.5e-5 | 1.000000 | 1.000000 / 2.0e-5 | 0.0 |
| 4 | 1764 | 229 | 1.000000 / 1.4e-4 | 1.000000 | 1.000000 / 2.1e-5 | 0.0 |
| en_zipformer1 | 1671 | 217 | 1.000000 / 1.1e-4 | 1.000000 | 1.000000 / 2.6e-5 | 0.0 |

Also verified: the cache-decode graph reproduces the prefill graph token-by-token
(`decode_graph_vs_prefill`, 4 teacher-forced steps, max_abs ≈ 1e-5, identical top-1 each step).

The `4.wav` / `en_zipformer1.wav` cases exercise **multi-window encoder attention** (229 / 217 valid rows =
`[104,104,21]` / `[104,104,9]`) and still match — so the 104-token window semantics are confirmed, not just the
single-window case.

---

## 10. Things that differ from the metadata / config / README (explicit)

1. **`use_rope:false` (`config.json:47`) is not implemented and must be ignored.**
   No module reads it; all three LM graphs contain `Cos`/`Sin`; the position-shift probe shows relative RoPE
   behaviour (§6.2). A port that omits RoPE will break.
2. **`model_bundle/preprocessor_config.json` is wrong/stale** — it says `feature_size: 80` and
   `return_attention_mask: false`, while the model needs 128 bins. The runtime sidesteps it by preferring
   `model_bundle/qwen3_asr_feature_extractor/preprocessor_config.json` (`asr_onnx_runtime.py:202-205`), which says
   128. Feeding 80 bins into `audio_hidden.onnx` is a shape error.
3. **`metadata.json` mixes two token counts.** `audio_hidden.max_audio_tokens = 390` and
   `valid_mask_sum_at_max = 390` describe the **encoder** output (rows of `audio_hidden`); the **prompt** at 30 s
   contains **375** `<|audio|>` placeholders, because the processor rule is floor-based (`§4.1`), and
   `tokens_per_chunk = 13` describes the encoder, not the prompt. Do not use 390 as the prompt count.
4. **`metadata.json:37` `vocab_size: 151652` vs `config.json:49` `vocab_size: 151936`.** Both are right for
   different objects: the tokenizer has 151652 entries, the embedding/logit matrix has 151936 rows (ids
   151652…151935 are never produced). Logits are 151936-wide.
5. **`chat_template.jinja` is never used.** It would add a `<|im_start|>system …` block, but the processor ignores
   `chat_template` and hand-builds the prompt (`processing_arkasr.py:296-297`), and the ONNX runtime never reads the
   file. Using the jinja template would change the prompt and degrade output.
6. **The HF reference needs two patches in this environment** (transformers 5.22.2 vs the 4.57.3 the checkpoint was
   written with). Both are silent — no exception, just garbage or slightly wrong audio:
   * `language_model.lm_head.weight` is reported **MISSING** and freshly initialised
     (`_tied_weights_keys` list form is not applied). Untied ⇒ garbage English/Chinese word salad. Fix:
     `model.language_model.lm_head.weight = model.language_model.model.embed_tokens.weight`.
   * `audio_encoder.positional_embedding.positional_embedding` is a **non-persistent buffer** and is **zero-filled**
     on load, so the audio tower silently loses the sinusoidal positions. Fix: recompute
     `[1500,896] = cat([sin, cos])` and copy it in.
   With both patches the transformers reference reproduces the bundle exactly (§9). `ref/run_transformers.py` applies
   them and prints `[pos-emb] …` / `[tie] …`.
7. **`processor_config.json` sets `audio_dtype: bfloat16`**, so `apply_chat_template` returns `input_features` in
   **bf16** and the HF path quantises the mel before the encoder. The ONNX path uses fp32. `run_transformers.py`
   has `--audio-dtype float32` if strict fp32 features are wanted (default keeps the released behaviour).
8. **Off-by-one in the runtime's `audio_feature_lengths`.** `encoder_feature_len = ceil(samples/160)`
   (`asr_onnx_runtime.py:258`) while the mel only has `floor(samples/160)` real frames. It is clipped to the tensor
   width, so it feeds `T+1` for any non-multiple-of-160 length. Effect: `E_valid` can be 1 higher than the HF
   value and the last pooled window mixes one extra (all-zero) frame; the **prompt** count is unaffected because it
   uses `sample_count` with the floor rule. Harmless in practice (all five wavs still match) but a port should use
   `floor`.
9. **`audio_hidden` rows at/after `valid_mask` are not zero** (`ref/parity.json` → `invalid_rows_are_zero: false`).
   They are real arithmetic on the zero-padded mel chunks. Always apply the mask; the valid rows are a **contiguous
   prefix** (verified `valid_mask_prefix_is_true` / `valid_mask_tail_is_false`), so a slice `hidden[:E_valid]` is
   equivalent.
10. **`max_total_len = 512` bounds the audio length.** `prompt_len + max_new_tokens ≤ 512`
    (`asr_onnx_runtime.py:627-632`). At 30 s the prompt is already 385 tokens, so `max_new_tokens=128` **raises**
    (`ValueError: … 385 + 128 > 512`; measured). Verified working with 112. For a sherpa-onnx port either raise the
    cache to ≥ 512 + a margin, or cap `max_new_tokens = 512 - prompt_len`.
11. **Audio longer than 30 s is truncated, not chunked** (`asr_onnx_runtime.py:246-247`). There is no long-form
    segmentation logic in the release.
12. **`metadata.json` `dummy_sequence_length: 16`** for the LM graphs is a tracing artefact only; the exported
    shapes are symbolic (`'sequence'`). `lm_logits.onnx` is the exception: its output is 2-D `[1, 151936]`.
13. The **README's default precision is int8/int8**; the fp32 graphs used here are the reference path. An int8 port
    must re-verify against these fp32 numbers (the int8 graphs are not part of this study).

---

## 11. What a sherpa-onnx C++ implementation needs

### 11.1 Audio front end
1. Read/resample to 16 kHz mono float32; **truncate at 30 s** (480000 samples).
2. Whisper log-mel exactly as §2.2: `hann(400, periodic)`, `rFFT(400)`, `hop 160`, `center=True` with
   **reflect** padding, **drop the last STFT frame**, power spectrum, 128-bin `slaney/slaney` mel over 0–8000 Hz,
   `log10(clamp 1e-10)`, `max(x, global_max-8)`, `(x+4)/4`. No dither, no mean/var normalisation.
3. Emit `float32[1,128,3000]`, true frames first, zero pad on the right.

### 11.2 Prompt building
1. `N_audio = max(1, ((L/160 + 1) / 2) / 4)` with integer floors (`L` in samples, capped at 480000).
2. Token sequence: `151647, 151648, [151646] * N_audio, 151649, 5501, 1356, 3114, 419, 7699, 13, 151650`.
   (The six text ids are fixed for the fixed English instruction; keep them as data, not as a re-tokenised string.)
3. Embedding lookup from `token_embedding.npy` (`[151936,512]`, fp32, row-major); replace the `N_audio`
   placeholder rows in order with the projected audio embeddings (`asr_onnx_runtime.py:300-310`).

### 11.3 Audio tower
1. 30 chunks × 100 frames reshape, 3× (conv k3 s2 p1 + exact-erf gelu), `conv_out` 7680→896.
2. `+ sin_table[0:13]` broadcast over chunks.
3. 18 pre-LN encoder layers, 14 heads × 64, scale `1/8` (equivalently `1/√64`), **block-diagonal attention with
   window 104 tokens** (8 chunks), no causal mask.
4. `ln_post` → `proj1` → gelu → `proj2` (896→1024) → 4 MLP-tower blocks (1024→4096→1024, pre-LN, GELU) → `final_norm`.
5. `E_valid = (T//100)*13 + ceil((T%100)/8)`, `T = floor(L/160)`; take the first `E_valid` rows.
6. `adaptive_avg_pool1d` (start `floor(i*E/N)`, end `ceil((i+1)*E/N)`) down to `N_audio`.
7. `LayerNorm(1024, eps 1e-5)` → `Linear(1024→512)` — this is `audio_projector.npz`.

> A cost optimisation: because attention is windowed at 104 tokens (= 8 chunks) and rows ≥ `E_valid` are ignored,
> only `ceil(E_valid/104)*8` chunks need to be executed. Running all 30 chunks reproduces the release bit-for-bit.

### 11.4 Decoder
1. Prefill the whole prompt in one causal pass; take the logits of the last position. (The `lm_cache_prefill` graph
   computes the full `[S,151936]` matrix — a port can restrict to the last position.)
2. Decode one token at a time with KV cache `[1,8,512,64]` fp32 per layer per K/V (16.78 MB),
   `position` incrementing, `attention_mask` = ones over `[0, position]` and zeros beyond, RoPE (θ=1e6) applied at
   the absolute `position`.
3. `logits[151646..151650] = -inf` before argmax; stop on 151645 `<|im_end|>` or 151643 `<|endoftext|>`;
   greedy; `max_new_tokens ≤ 512 - prompt_len` (≈ 112–127 at 30 s).

### 11.5 Detokenisation
Qwen2 byte-level BPE; `skip_special_tokens=False` then strip `<|...|>` tokens and leading punctuation/whitespace
(§7). `<|audio|>` is not flagged special, so filter it explicitly if the -inf blocking is dropped.

### 11.6 Limits and budget

| item | value |
|---|---|
| max audio | 30 s (hard truncate) |
| prompt tokens | `N_audio + 10`, `N_audio ≤ 375` → `prompt_len ≤ 385` |
| cache | `max_total_len 512` → `prompt_len + max_new ≤ 512` |
| audio tower weights | 219.9 M params = 879.8 MB fp32 (int8 graph: 234.7 MB) |
| LM weights | 103.5 M params = 414.0 MB fp32 ×3 graphs, or 1× (prefill+decode share nothing) |
| embedding table | 77.8 M params = 311.2 MB fp32 |
| KV cache | 16.78 MB |
| logits buffer | 151936 × 4 B = 607.7 kB |
| measured RTF (ORT CPU, M4 Pro, 1 thread pool) | 0.086–0.257 for 5–18 s audio |

---

## 12. Reproduce

```bash
cd /Users/yunyu/Documents/DSH/AX-model/audio8
P=/Users/yunyu/Documents/DSH/AX-model/.venv/bin/python
export HF_HOME=$PWD/.hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

$P ref/inspect_graphs.py                       # -> ref/onnx_io.json   (graph IO table)
$P ref/run_transformers.py downloaded/wavs/en.wav downloaded/wavs/zh.wav \
     downloaded/wavs/0.wav downloaded/wavs/4.wav downloaded/wavs/en_zipformer1.wav \
     --out ref/hf_runs.json                    # -> ref/hf_runs.json   (task 5)
$P ref/run_bundle.py --out ref/transcripts.json --hf-runs ref/hf_runs.json \
     downloaded/wavs/en.wav downloaded/wavs/zh.wav \
     downloaded/wavs/0.wav downloaded/wavs/4.wav downloaded/wavs/en_zipformer1.wav
                                               # -> ref/transcripts.json + ref/tensors/*.npz (task 4)
$P ref/parity.py en zh 0 4 en_zipformer1       # -> ref/parity.json    (numerical parity)
```

Download of the two repos (mirror):

```python
from huggingface_hub import snapshot_download
snapshot_download("Edge0/Audio8-ASR-0.1B", local_dir="downloaded/hf_model")
snapshot_download("Edge0/Audio8-ASR-0.1B-onnx-runtime", local_dir="downloaded/onnx_release",
                  ignore_patterns=["*_int8.onnx", "*_int4.onnx", "*_int8.onnx.data", "*_int4.onnx.data",
                                   "audio_hidden_int8.onnx", "static/*"])
```

Test audio: `csukuangfj/2026-06-03` (ModelScope, `0.wav`…`4.wav`, en+zh mixed),
`csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/{en,zh}.wav`,
`csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/1.wav`.

## 13. Files produced

| file | content |
|---|---|
| `reports/01-model-spec.md` | this spec |
| `ref/run_bundle.py` | official-engine runner with tensor capture |
| `ref/run_transformers.py` | transformers reference (with the two required patches) |
| `ref/parity.py` | ONNX↔transformers parity + RoPE probe + decode-graph probe |
| `ref/inspect_graphs.py` | ONNX IO table generator |
| `ref/transcripts.json` | bundle vs transformers transcripts + timings (5 wavs) |
| `ref/hf_runs.json` | transformers-only detail (prompt ids, mel shape, encoder length, tokens) |
| `ref/parity.json` | all parity numbers quoted above |
| `ref/onnx_io.json` | full IO/op table for the four graphs |
| `ref/tensors/*.npz` | mel, prompt ids, inputs_embeds, audio embeddings, hidden, mask, audio |
| `ref/tensors_hf/*.npz` | transformers mel / prompt ids / generated ids |
