# 02 — sherpa-onnx 新增 `audio8-asr` 离线模型类型：集成调研与实施计划

- 调研对象：`/tmp/sherpa-onnx`（HEAD `031a93a`；`csrc` 与 `master` `60847b8` 无差异；`x-asr-axera` 分支与本主题无关）
- 参考实现：`OfflineQwen3ASRModel` / `OfflineRecognizerQwen3ASRImpl`
- 目标模型：Audio8-ASR-0.1B
- **实施目标契约：`audio8/exported/{encoder.onnx, decoder.onnx}`（2 图）** —— 本仓库导出侧已产出并通过 CER=0.000 验证，见 `03-export.md`。本报告的 checklist 以这份契约的**实测 I/O 签名**为准（§2.2）。
- 本轮实测结论：
  - tokenizer **不用改**：`QwenAsrTokenizer` 直接读 Audio8 的 `hf_model/`，产出的 id 与 HF `tokenizers` **逐位相同**（§3，C++ 实跑）。
  - macOS 最小 configure **成功**（493 s，几乎全耗在下载 onnxruntime 1.28.2），`sherpa-onnx-core` 全量编译 **36.9 s**，`qwen3-asr-cxx-api` **4.9 s** 产出 31 MB 可执行文件（§6）。
  - 与 Qwen3-ASR 的差异集中在 **4 处硬点**：KV 布局 `[B,H,L,D]`、定长 `attention_mask[1,512]`、mel 必须补零到 3000 帧 + `num_rows`、prompt/stop/block-id 各不相同（§6）。

> ⚠️ 协作事实：`/tmp/sherpa-onnx` 在本轮调研期间被**另一个 agent** 写入了两个未跟踪文件（`sherpa-onnx/csrc/offline-audio8-asr-model-config.{h,cc}`，mtime 19:27，作者署名 `yuyun2000`）。本会话对该 checkout **全程只读**，未创建/修改任何文件；构建产物全部在 `/tmp/sherpa-survey-work/`。§5 的 checklist 已按该文件的命名与字段对齐，并列出它当前**缺的校验项**（§5.1.1）。

---

## 0. 一句话流程

```
OfflineRecognizerConfig → OfflineRecognizerImpl::Create()
  → 靠「哪个子 config 非空」分派（没有 model-type 枚举）
  → OfflineRecognizerAudio8AsrImpl
       CreateStream(): WhisperTag{dim=128, align_to_stft_center=true}
       Decode(): GetFrames() → 截到 3000 帧 → 补零 → NormalizeWhisperFeatures
                 → keep_rows = 编码器有效行数 = full*13 + after_cnn(rem)
                 → ForwardEncoder(mel[1,128,3000], keep_rows) → hidden[1,390,1024]
                 → hidden[:keep_rows] 池化到 K = ((frames+1)//2)//4 个音频 token
                 → audio_hidden[1,K,1024]
                 → GenerateText(): prompt(N+10) + cache([1,8,512,64]×8)
                    → ForwardLLM(prefill, mask[1,512]) → ApplyKvDeltaInplace
                    → 每步 1 token 循环 → tokenizer->Decode → OfflineRecognitionResult
```

详细调用图见 `02-qwen3-asr-flow.md`（Qwen3 路径，Audio8 只换 §5 列出的部件）。

---

## 1. Qwen3-ASR 集成端到端解剖

### 1.1 文件清单与职责

| 文件 | 行数 | 职责 |
|---|---|---|
| `sherpa-onnx/csrc/offline-qwen3-asr-model-config.h` | 73 | `OfflineQwen3ASRModelConfig`（4 路径 + 6 解码参数） |
| `sherpa-onnx/csrc/offline-qwen3-asr-model-config.cc` | 154 | `Register()`（CLI 键）、`Validate()`、`ToString()` |
| `sherpa-onnx/csrc/offline-qwen3-asr-model.h` | 101 | 3 图模型类声明 |
| `sherpa-onnx/csrc/offline-qwen3-asr-model.cc` | 859 | 3 个 `Ort::Session`、IO 名字缓存、KV 分配与 delta 拷贝 |
| `sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h` | 87 | recognizer 声明 + 暴露 `TrimAudioFeatures()` 供单测 |
| `sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.cc` | 1248 | 特征规整、prompt 构造、采样、greedy 循环、退化截断、结果组装 |
| `sherpa-onnx/csrc/qwen-asr-tokenizer.{h,cc}` | 62/1339 | Qwen BPE（vocab.json + merges.txt + tokenizer_config.json）**直接复用** |
| `offline-recognizer-qwen3-asr-impl-test.cc` | — | `TrimAudioFeatures` 的 3 个 gtest（不需要权重） |

### 1.2 注册与分派：**没有 model type 枚举**

1. `OfflineModelConfig`（`offline-model-config.h:30-125`）把所有模型族平铺成字段（如 `:48` `qwen3_asr;`）。
2. `OfflineModelConfig::Register()`（`offline-model-config.cc:14-65`）逐个调子 config 的 `Register`，`qwen3_asr.Register(po)` 在 `:32` —— **CLI 键名在这一层定**。
3. `OfflineModelConfig::Validate()`（`:67-225`）：
   - `:103` **特例**：`if (funasr_nano.encoder_adaptor.empty() && qwen3_asr.conv_frontend.empty())` 才检查 `tokens` 文件。**Audio8 必须加进这个条件**，否则 `--tokens` 缺失即失败。
   - `:186-188` 分派到 `qwen3_asr.Validate()`；顺序在 `telespeech_ctc`/`transducer` 之前。
4. `OfflineRecognizerImpl::Create(config)`（`offline-recognizer-impl.cc:249-251`）按「`conv_frontend` 非空」分派；manager 版在 `:657-659`。**两处都要加。**
5. `model_type`（`offline-model-config.cc:48-52`）只是「省一次加载」的提示，新模型不必登记。

### 1.3 CLI 键与校验（`offline-qwen3-asr-model-config.cc:15-134`）

```
--qwen3-asr-conv-frontend  --qwen3-asr-encoder  --qwen3-asr-decoder
--qwen3-asr-tokenizer      --qwen3-asr-hotwords
--qwen3-asr-max-total-len  --qwen3-asr-max-new-tokens
--qwen3-asr-temperature    --qwen3-asr-top-p     --qwen3-asr-seed
```
`Validate()` 要求 tokenizer 是**目录**且含 `vocab.json` / `merges.txt` / **`tokenizer_config.json`**（`:87-107`）。

### 1.4 特征计算：Whisper 风格 log-mel（128 bin）

- `CreateStream()`（`impl.cc:428-437`）：
  ```cpp
  WhisperTag tag;
  tag.dim = kQwen3MelDim;             // 128, impl.cc:47
  tag.align_to_stft_center = true;    // impl.cc:435
  return std::make_unique<OfflineStream>(tag);
  ```
  **不用** `config.feat_config`；采样率固定 16 kHz。
- `WhisperTag`（`offline-stream.h:61-74`）：`align_to_stft_center` 让分帧中心对齐 Whisper 的 `torch.stft(center=True)`，否则晚 5 ms；Qwen3-ASR 对此敏感（上游 issue #3535）。
- 实现：`offline-stream.cc:82-107`（构造 `knf::OnlineWhisperFbank`，`whisper_center_padding_ = 80` 采样点）、`:232-260`（`GetFrames()` 截到 `wav_len/160` 帧）。
- kaldi-native-fbank v1.23.0 `WhisperFeatureOptions` 明确：**16 kHz / 25 ms 窗 / 10 ms 移 / hann / snip_edges=false** ⇒ `n_fft=400, hop=160`，librosa（slaney）mel。
- 归一化：`impl.cc:1184` → `NormalizeWhisperFeatures()`（`math.cc:115-135`）：`log10(clamp(x,1e-10))` → `max(·, max-8)` → `(x+4)/4`，与 HF 一致。
- 张量：`GetFrames()` 返回**一维** `num_frames×128`，识别器自行 `CreateTensor<float>` 成 `[1,T,128]`（`impl.cc:1189-1194`）。Qwen3 路径 **T 动态、不 pad**。

### 1.5 三段图契约（`offline-qwen3-asr-model.cc`）

| 图 | 输入 | 输出 | 校验 |
|---|---|---|---|
| conv_frontend | 1 个：`[1,T,128]` f32 | 1 个：`[1,T',C]` | 各 1（`:229-239`） |
| encoder | 2 个：`conv_out` + `feature_attention_mask`（**bool** `[1,T']`） | 1 个：`[1,A,H]` | 2 进 1 出（`:261-272`）；mask 必须 bool（`:376-385`） |
| decoder | `4+2L`：`input_ids[1,T]`、`audio_features[1,A,H]`、`attention_mask[1,T]`、`cache_position[T]`、KV 对 | `1+2L`：`logits[1,T,V]` + KV delta | `(nout-1)%2==0`（`:293-300`）；`nin==4+2L`（`:302-309`） |

audio token 数用解析式：`FeatToAudioTokensLen(feat_len, 100)`（`impl.cc:77-108`）＝ 3 次 stride-2 conv，`100 帧 → 13 token`。

### 1.6 文本解码（`impl.cc`）

- **Tokenizer**：`QwenAsrTokenizer`（纯 C++ Qwen BPE）。`ParseAddedTokens()`（`qwen-asr-tokenizer.cc:838-899`）读 `tokenizer_config.json` 的 `added_tokens_decoder`；`IsSpecialToken()`（`:768-778`）匹配 `<|...|>`；`BuildSpecialTokens()` 按长度降序；`Encode()`（`:1202-1244`）先最长匹配特殊 token 再做 `SplitByQwen3Pattern` + `ByteLevelEncode` + BPE；`Decode()`（`:1255-1284`）跳过 `<|im_start|>/<|im_end|>`，其它特殊 token **原样输出**。
- **prompt**（`:439-507`）：`audio_pad_ids_ = Encode("<|audio_pad|>")`；`prompt_ids_after_ = Encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")`；`BuildSourceIds()` = `"<|im_start|>system\n"+hotwords+"<|im_end|>\n<|im_start|>user\n<|audio_start|>"` + `audio_pad_ids_ × A` + after（可选 `language X` + `<asr_text>`）。
- **采样**：`temperature<=1e-6` 走 argmax（`:593-627`），否则 softmax+top-p+`mt19937`（`rng_mutex_` 保护，因为 `DecodeStreams` 可被多线程并发调用，`:76-82`）。
- **greedy 循环**（`:765-1147`）：prefill 取 `logits[ctx-1]`；首 token==eos 时以 `avoid_id=eos` 重采，仍 eos → 空；循环 `step 1..max_new_tokens-1`，`cur_len>=max_seq_len` break；末尾 64 token 内 distinct≤3 判死循环并截断（`:326-350`、`:1089-1102`）；剥 `language X<asr_text>` 前缀（`:1105-1123`）；`result.text=Decode(ids)` + 删 U+FFFD（`:1125-1126`）；`result.tokens` 用 `GetTokenStringStreaming` 逐 token 拼。
- 回到 `Decode()`：`ApplyHomophoneReplacer`（`:1233`）→ `SetResult`；最外层 `OfflineRecognizer::DecodeStreams`（`offline-recognizer.cc:173-181`）再 `RemoveLeadingSpaces`。
- `OfflineRecognitionResult` 只填 `text` + `tokens`，**无时间戳**。

### 1.7 KV cache

- 容量：`max_total_len_` **优先取 decoder 第一个 past_key 输入静态 shape 的第 1 维**（`offline-qwen3-asr-model.cc:311-318`），动态则退回 `--qwen3-asr-max-total-len`。
- `CreateEmptyKVCache(batch)`（`:533-581`）：`tpl[2]=kv_h`、`tpl[3]=hd`，分配 `[batch, max_total_len_, kv_h, hd]`，类型取图输入（f32/f16/u16），置 0。**假设 `[B,L,H,D]`**。
- `ApplyKvDeltaInplace()`（`:583-749`）：`pos0=cache_position[0]`，`pos0+S>max_total_len_` 时**截断 S**（打日志不报错）；逐层 `memcpy`，目标偏移 `(b*max_total_len_+pos0)*(kv_h*hd*bytes)`。第 1 维是序列的假设在 `:693`、`:720-736` 再次硬编码。

---

## 2. Audio8-ASR 契约

### 2.1 官方 bundle（`AutoArk-AI/Audio8-ASR-0.1B-onnx-runtime`）—— **仅作背景，不采用**

5 张图（`audio_hidden` / `lm_cache_prefill` / `lm_cache_decode` / `lm_logits` + host 侧 `.npy/.npz` 权重），decoder 吃 `inputs_embeds`（token embedding 在 host，fp32 311 MB），prefill/decode 两张图，KV `[1,8,512,64]`。**不采用**，因为 host 侧 311 MB embedding + 双图对边缘部署不友好，且与本仓库已导出的 2 图契约冲突。

### 2.2 **实施目标**：本仓库 `audio8/exported/` 的 2 图契约（实测 I/O）

用 `onnxruntime 1.23.2` 直接 dump 的签名（`audio8/exported/`，fp32，`metadata.json` 见下）：

```jsonc
// exported/metadata.json
{ "frames":3000, "enc_tokens":390, "max_total_len":512, "num_layers":8,
  "num_heads":8, "num_kv_heads":8, "head_dim":64, "hidden_size":512,
  "audio_token_id":151646, "vocab_size":151936 }
```

```
=== encoder.onnx
  in : audios    tensor(float) [1, 128, 3000]      # Whisper log-mel，零填充到 3000 帧
  in : num_rows  tensor(int64) [1]                 # 编码器有效行数（不是帧数、也不是 prompt token 数）
  out: hidden    tensor(float) [1, 390, 1024]      # 第 num_rows 行之后无效

=== decoder.onnx
  in : input_ids        tensor(int64) [1, 'seq_len']
  in : audio_hidden     tensor(float) [1, 'num_audio', 1024]   # = pool(hidden[:num_rows] → K)
  in : attention_mask   tensor(int64) [1, 512]                 # ★ 定长 = max_total_len
  in : cache_position   tensor(int64) ['seq_len']
  in : cache_key_i / cache_value_i   tensor(float) [1, 8, 512, 64]   i = 0..7
  out: logits           tensor(float) [1, 'seq_len', 151936]
  out: present_key_i / present_value_i tensor(float) [1, 8, 'seq_len', 64]   # 只回「本次新增」
```

语义（`03-export.md` + `export/onnx_runtime.py` 实测）：

| 项 | 值 |
|---|---|
| encoder 的 `num_rows` | **编码器有效行数**（不是 prompt 的 token 数！）：`keep_rows = min(full*13 + after_cnn(rem), 390)`，其中 `full = frames/100`、`rem = frames%100`、`after_cnn(x)` 是 3 次 `(x-1)//2+1`。30 s → 390；10.05 s（1005 帧）→ 10*13+after_cnn(5)=130+1=131 |
| prompt 的音频 token 数 `K` | `max(1, ((mel_frames + 1) // 2) // 4)`，`mel_frames = floor(num_samples/160)`。30 s → 375；10.05 s → 125 |
| 两者的桥接 | host 侧把 `hidden[:keep_rows]` 做 `adaptive_avg_pool` 到 `K` 行（`F.adaptive_avg_pool1d` 语义），再作为 `audio_hidden[1,K,1024]` 喂 decoder。**这一步不能省**：131≠125、390≠375 |
| 30 s 上限 | 图固定 3000 帧 / 390 token；**更长的音频必须先截到 480000 采样点**，否则 `K > 390` 越界 |
| prompt | `<|user|><|begin_of_audio|>` + `<|audio|>`×K + `<|end_of_audio|>Please transcribe this audio.<|assistant|>`；**无 system、无 BOS**；`prompt_len = K + 10` |
| KV 布局 | **`[B, H, L, D]`**（序列在第 2 维）—— 与 Qwen3 的 `[B, L, H, D]` 相反 |
| KV 写回 | `present_*` 只有 `seq_len` 行（= 新增），按 `cache_position` 写回 |
| attention_mask | **定长 512**，前 `valid_len` 个置 1，其余 0 |
| 投影层 | `LayerNorm + Linear(1024→512)` **在 decoder 图里**，所以 `audio_hidden` 是 1024 维 |
| 结束符 | 151645（`<|im_end|>`），另需把 151643（`<|endoftext|>`/pad）也当 stop（官方参考实现如此） |
| 屏蔽 id | 151646,151647,151648,151649,151650（`extra_block_token_ids`） |
| 图大小 | `encoder.onnx` 880 MB + `decoder.onnx` 727 MB（磁盘；fp32） |

**`K` 公式对 5 条参考音频全部命中**（用 `03-export.md` 的时长 + `onnx-reference.json` 的 `audio_tokens` 交叉验证，可作 §5.9 单测的黄金值）：

| wav | 时长 | 采样点 | mel 帧 | `((帧+1)/2)/4` | 实测 `audio_tokens` | `prompt_len`(=K+10) |
|---|---|---|---|---|---|---|
| 0.wav | 10.05 s | 160800 | 1005 | 125 | **125** | 135 ✓ |
| 1.wav | 5.10 s | 81600 | 510 | 63 | **63** | 73 ✓ |
| 2.wav | 4.69 s | 75040 | 469 | 58 | **58** | 68 ✓ |
| 3.wav | 8.83 s | 141280 | 883 | 110 | **110** | 120 ✓ |
| 4.wav | 17.64 s | 282240 | 1764 | 220 | **220** | 230 ✓ |

> 结论：decoder 的输入/输出**结构**与 Qwen3 完全同形（`4+2L` 进 / `1+2L` 出），所以 `OfflineQwen3ASRModel` 的骨架可以照抄；变的是**每个张量的语义**（mask 定长、KV 布局、audio 维度 1024）以及 encoder 换成「mel + num_rows」。

---

## 3. Tokenizer 可行性 —— **已用 C++ 实测确认，零改动**

把 `qwen-asr-tokenizer.cc` + `file-utils.cc` + `text-utils.cc` 单独编成可执行文件，直接读 Audio8 的 `hf_model/` 跑：

```
audio <|audio|> id  = 151646      <|user|> id        = 151647
<|assistant|> id    = 151650      <|begin_of_audio|> = 151648
<|end_of_audio|>    = 151649      <|im_end|> id      = 151645
<|endoftext|> id    = 151643      eos = 151645       pad = 151643
scaffold ids (11): 151647 151648 151646 151649 5501 1356 3114 419 7699 13 151650
prompt len = 385, audio ids = 375        ← 与 N+10 规则一致（N=375 = 30 s）
```

与 Python `tokenizers 0.23.3` **逐位相同**（含 `5501 1356 3114 419 7699 13` = "Please transcribe this audio."）。原因：

| 检查项 | 结果 |
|---|---|
| `vocab.json` | 151643 条，与 `tokenizer.json.model.vocab` **完全相等**（id 0–151642） |
| `merges.txt` | 151388 行（首行 `#version: 0.2`，`ParseMerges` 跳过 `#`），与 `tokenizer.json` 一致 |
| `added_tokens_decoder` | 151643–151651 共 9 个 → `ParseAddedTokens()` 全部登记 ✔ |
| 预分词正则 | 与 Qwen3 **逐字相同**，`SplitByQwen3Pattern()` 适用 |
| `GetEosTokenId()` / `GetPadTokenId()` | 151645 / 151643 ✔ 正好是 Audio8 的 stop 集合 |
| `Encode` / `Decode` | 与 HF 一致；但 `Decode` 会把 `<|audio|>` 等**原样输出**（实测），故必须做 block-id + 后过滤 |

复现命令（utf8proc 符号被加了前缀，必须带 31 个 `-D`，见 `cmake/utf8proc.cmake:60-91`）：

```bash
DEFS=$(sed -n '61,96p' cmake/utf8proc.cmake \
  | grep -oE "utf8proc_[A-Za-z_0-9]+=sherpa_onnx_utf8proc_[A-Za-z_0-9]+" \
  | sed 's/^/-D/' | tr '\n' ' ')
clang++ -std=c++17 -O1 $DEFS -I$SRC -I$BUILD/_deps/json-src/include \
  -I$BUILD/_deps/utf8proc-src -o tokcheck tokcheck.cc \
  $SRC/sherpa-onnx/csrc/{qwen-asr-tokenizer.cc,file-utils.cc,text-utils.cc} \
  $BUILD/lib/libutf8proc.a
```

---

## 4. 复用矩阵（能抄 / 必须改）

| Qwen3 部件 | Audio8 能否复用 | 说明 |
|---|---|---|
| `QwenAsrTokenizer` | ✅ **原样** | §3 实测 |
| `QwenAsrTokenizer::Decode` / `GetTokenStringStreaming` | ✅ 原样 | 但要额外过滤特殊 token |
| `TrimAudioFeatures` | ✅ 原样（改名） | 尾部静音裁剪 + 全静音短路仍然有用 |
| `FeatToAudioTokensLen` | ❌ **换成** `((frames+1)/2)/4` | 30 s 时 390 vs 375，不能用 conv 口径 |
| 排序/采样函数（`SampleToken*`、`IsDegenerateRepetition`、`TrimDegenerateTail`、`RemoveUtf8ReplacementChars`） | ✅ 原样 | 纯逻辑 |
| greedy 主循环骨架 | ✅ 照抄 | 只改 mask 定长、eos 集合、block ids |
| `CreateStream()`（WhisperTag 128 + center） | ✅ 原样 | 参数完全相同 |
| `NormalizeWhisperFeatures` | ✅ 原样 | 但要在**归一化之后**补零到 3000 帧 |
| 3 图 `OfflineQwen3ASRModel` | ⚠️ **骨架可抄，语义要改** | 少一张图；encoder 输入/语义不同；KV 布局相反；mask 定长 |
| `CreateEmptyKVCache` | ❌ **必须重写** | `[B,H,L,D]` |
| `ApplyKvDeltaInplace` | ❌ **必须重写** | 同上 |
| `InitPromptTemplateIds` / `BuildSourceIds` | ❌ **必须重写** | 模板、`<|audio|>` 单 id、无 `<asr_text>`、无 language、无 hotword 注入 |
| `ApplyHomophoneReplacer` | ✅ 保留 | 基类能力 |

---

## 5. 文件级 checklist（`audio8-asr`）

> 命名与字段已对齐另一个 agent 已写下的 `OfflineAudio8AsrModelConfig`（`encoder` / `decoder` / `tokenizer` / `max_total_len` / `max_new_tokens`）。`$SRC = /tmp/sherpa-onnx`。

### 5.1 config（已存在，需补）

`sherpa-onnx/csrc/offline-audio8-asr-model-config.{h,cc}`（已由另一 agent 创建，约 1.6 KB + 2.9 KB）

#### 5.1.1 必须补的 3 处（当前版本有**静默失败**风险）

1. **`Validate()` 缺 `tokenizer_config.json` 检查**（现在只查 `vocab.json` + `merges.txt`）。
   `QwenAsrTokenizer::Init()`（`qwen-asr-tokenizer.cc:1183-1197`）会读 `tokenizer_config.json`；文件缺失时 `ReadTextFile` 返回空 → `ParseAddedTokens()` **静默 return** → 151646/151647/…/151651 全部丢失 → `<|audio|>` 编码成 BPE 碎片 → prompt 全错，且**不会报错**。这是最容易踩的坑。
   ```cpp
   if (!FileExists(tokenizer + "/tokenizer_config.json")) { SHERPA_ONNX_LOGE(...); return false; }
   ```
2. **`max_total_len` 应由图推断**，不能只信 CLI（见 5.2 的 `InitDecoderSession`）。CLI 值只作 fallback。
3. 建议加 `temperature`/`top_p`/`seed`（若确定只用 greedy，可保留常量 `1e-6`，但请在注释里写明「Audio8 只有 greedy」）。

#### 5.1.2 可选

- `hotwords`（Audio8 是 logit boosting，见 §6 Q9）—— 首版可只接受参数并 warning。

### 5.2 新建模型类

**`sherpa-onnx/csrc/offline-audio8-asr-model.h`**（copy `offline-qwen3-asr-model.h` 后删掉 conv 相关）

```cpp
class OfflineAudio8AsrModel {
 public:
  explicit OfflineAudio8AsrModel(const OfflineModelConfig &config);
  template <typename Manager> OfflineAudio8AsrModel(Manager *mgr, const OfflineModelConfig &config);
  ~OfflineAudio8AsrModel();

  // mel [1,128,3000] f32 + num_rows int64 [1] → hidden [1,390,1024]
  Ort::Value ForwardEncoder(Ort::Value audios, Ort::Value num_rows);

  // input_ids[1,T] + audio_hidden[1,K,1024] + attention_mask[1,512]
  // + cache_position[T] + KV[L] → (logits, KV delta)
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLM(Ort::Value input_ids, Ort::Value audio_hidden,
             Ort::Value attention_mask, const Ort::Value &cache_position,
             const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv);

  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(int64_t batch);
  void ApplyKvDeltaInplace(std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
                           const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
                           const Ort::Value &cache_position);
  int32_t GetMaxTotalLen() const;
  OrtAllocator *Allocator() const;
 private:
  class Impl; std::unique_ptr<Impl> impl_;
};
```

**`offline-audio8-asr-model.cc`**（从 `offline-qwen3-asr-model.cc` 改造，最小 diff 形状）：

| 原位置 | 改动 |
|---|---|
| `:38-39` 常量 | 保留 `kDecoderCachePositionInputIndex=3`、`kDecoderPastKvInputStartIndex=4`（**与 Audio8 的图完全一致**） |
| `:121-195` `Impl` ctor | 拆掉 `conv_sess_`（3 处 → 2 处）：只 `encoder_sess_` + `decoder_sess_`；`Ort::Env` 名改 `"audio8-asr"` |
| `:215-272` `InitConvFrontend` / `InitEncoderSession` | 删掉 conv 版；`InitEncoderSession` 改成**要求 2 进 1 出**（`audios`、`num_rows`）+ 校验第 0 个输入是 `[1,128,3000]`（若第 2 维为 -1 则退化为动态、记 `frames_padded_=0`） |
| `:288-344` `InitDecoderSession` | **关键改动**：<br>① `num_layers_ = (nout-1)/2` 保留（=8）<br>② `nin == 4+2*num_layers_` 保留（=20）<br>③ **`max_total_len_` 取 `cache_key_0` 的 `shape[2]`**（不是 `[1]`！`[1,8,512,64]` → `tpl[2]=512`）<br>④ 新增 `kv_h_ = tpl[1]`、`hd_ = tpl[3]`<br>⑤ 新增「从 `attention_mask` 输入 shape 推出 `mask_len_ = 512`」<br>⑥ 断言 `mask_len_ == max_total_len_` |
| `:347-413` `ForwardConvFrontend` + `ForwardEncoder` | conv 删除；`ForwardEncoder(audios, num_rows)` 2 入 1 出，**不再检查 bool 类型**（`num_rows` 是 int64），改检查 int64 |
| `:415-531` `ForwardLLM` | 保留结构；把第 1 个参数语义从 `input_ids+audio_features` 换成 `input_ids+audio_hidden`（调用形状相同，无需改代码，但注释要改）；`audio_features` 的最后一维断言 `== 1024` 更安全 |
| `:533-581` `CreateEmptyKVCache` | **重写成 `[batch, kv_h_, max_total_len_, hd_]`**（顺序是 `[B,H,L,D]`） |
| `:583-749` `ApplyKvDeltaInplace` | **重写偏移**：<br>`elem_bytes` 同上；每 token 字节数 `kv_h_*hd_*elem` 不变；<br>目标偏移 `((b*kv_h_ + h)*max_total_len_ + pos0) * (hd_*elem)`；<br>源偏移 `((b*kv_h_ + h)*S + 0) * (hd_*elem)`；<br>`S = dk_shape[2]`（第 2 维才是序列）；<br>逐 `b`、逐 `h` 两层循环（或把 `(b,h)` 合并成一次 `memcpy` 当 `S` 覆盖连续区间） |
| `:751-752` `GetMaxTotalLen` / `Allocator` | 不变 |
| `:849-857` 模板显式实例化 | 保留 |

> 提示：`ApplyKvDeltaInplace` 在 `S == max_total_len_` 且 `pos0 == 0` 时（prefill 常见），`[B,H,L,D]` 下「每 (b,h) 的一段连续 L×D」正好等价于整块顺序拷贝 —— 可以先写通用循环，后续再优化。

### 5.3 新建 recognizer impl

**`sherpa-onnx/csrc/offline-recognizer-audio8-asr-impl.h/.cc`**（copy `offline-recognizer-qwen3-asr-impl.{h,cc}`）

改动清单（按原文件位置）：

| 原位置 | 改动 |
|---|---|
| `impl.cc:43` `kQwen3ChunkSize=100` | **删除**（改用 merge_factor 规则），或保留作 `13 tokens/100 帧` 的注释常量 |
| `impl.cc:47` `kQwen3MelDim=128` | 保留 → `kAudio8MelDim` |
| 新增常量 | `kAudio8Frames = 3000`；`kAudio8MaxTokens = 390`；`kAudio8MergeFactor = 4`；`kAudio8StopIds = {151645, 151643}`；`kAudio8BlockIds = {151646,151647,151648,151649,151650}` |
| `impl.cc:50-52` prompt 常量 | `kAudio8PromptPrefix = "<|user|><|begin_of_audio|>"`；`kAudio8PromptSuffix = "<|end_of_audio|>Please transcribe this audio.<|assistant|>"` |
| `impl.cc:56-59` `Qwen3FormatHotwordsForPrompt` | 删除（hotword 不注入） |
| `impl.cc:77-108` `FeatToAudioTokensLen` | **删除**，改成**两个**函数：<br>① `Audio8EncoderRows(frames)` = `full*13 + after_cnn(rem)`，`min(·, 390)` → 传 encoder 的 `num_rows`；<br>② `Audio8AudioTokenCount(frames)` = `max(1, ((frames+1)/2)/4)` → prompt 里 `<|audio|>` 的个数；<br>③ `Audio8AdaptiveAvgPool(in, num_in_rows, dim, num_out_rows, out)`（`floor/ceil` 语义同 `F.adaptive_avg_pool1d`） |
| `impl.cc:129-207` `TrimAudioFeatures` | 保留（改名） |
| `impl.cc:321-389` 退化重复 | 保留 |
| `impl.cc:439-457` `InitPromptTemplateIds` | `audio_id_ = tokenizer_->GetTokenId("<|audio|>")`（**单 id**，断言==151646 或至少 >=0）；`prefix_ids_ = Encode(kAudio8PromptPrefix)`；`suffix_ids_ = Encode(kAudio8PromptSuffix)`；<br>**删除** `asr_text` / `audio_pad_ids_` / `prompt_ids_after_` 之间的拼接逻辑；<br>**实测**: `Encode(prefix)=={151647,151648}`、`Encode(suffix)=={151649,5501,1356,3114,419,7699,13,151650}`（可直接硬编码校验，做单测） |
| `impl.cc:459-507` `BuildSourceIds` | 重写为 `prefix_ids_ + audio_id_×K + suffix_ids_`；删 language / hotwords 分支 |
| `impl.cc:544-763` 采样 | 保留；在 `SampleTokenWithTemperatureAndTopP` **之前**加 `MaskBlockedIds(row, vocab)`（把 block ids 置 `-inf`），或直接在选 argmax 的循环里 `if (IsBlocked(i)) continue;` |
| `impl.cc:765-1147` `GenerateText` | 保留骨架；<br>① 新增 `mask_len_ = model_->GetMaskLen()`；prefill mask = `[1, mask_len_]`，前 `context_len` 置 1（**不是** `[1,context_len]`，`impl.cc:946-950` 必须改）；<br>② 每步 decode mask = `[1, mask_len_]`，前 `cur_len+1` 置 1（**不是** `[1,1]`，`impl.cc:1048-1052` 必须改）；<br>③ `eos_id` → `IsStopId(next_id)`（两处：`:990`、`:1082`）；<br>④ 删 `asr_text` 前缀剥离（`:1105-1123`）；<br>⑤ 文本后过滤：截断 `<|im_end|>` 之后、删 `<|...|>` 特殊 token、`std::regex`-free 空白压缩；<br>⑥ `max_new_tokens` clamp 到 `max_seq_len - context_len`（见 Q7） |
| `impl.cc:1156-1236` `Decode` | **重写特征段**：<br>`f = GetFrames()` → `num_frames = f.size()/128`<br>→ **截到 3000 帧**（`num_frames = min(num_frames, 3000)`，超出时 `SHERPA_ONNX_LOGE` 警告 "audio truncated to 30 s"）<br>→ `NormalizeWhisperFeatures(f, num_frames, 128)`（**只归一化真实帧**）<br>→ 分配 `[1,128,3000]` f32，**零填充**，把 `num_frames×128` 拷进去<br>→ `keep_rows = Audio8EncoderRows(num_frames)`；`K = Audio8AudioTokenCount(num_frames)`<br>→ `ForwardEncoder(mel, keep_rows)` → `hidden[1,390,1024]`<br>→ `Audio8AdaptiveAvgPool(hidden, keep_rows, 1024, K, buf)` → `audio_hidden[1,K,1024]`<br>→ `GenerateText(audio_hidden, K, stream)` |
| `impl.cc:1233` `ApplyHomophoneReplacer` | 保留 |

### 5.4 注册与分派（4 处，必改）

| 文件 | 位置 | 改动 |
|---|---|---|
| `offline-model-config.h` | `:16` include；`:48` 字段；`:88`/`:111` ctor | 加 include、`OfflineAudio8AsrModelConfig audio8_asr;`、ctor 形参（**顺序必须与 python `py::init` 同步**） |
| `offline-model-config.cc` | `:32` Register；`:103` tokens 特例；`:186` Validate；`:248` ToString | `audio8_asr.Register(po);`；`... && audio8_asr.encoder.empty()`；`if (!audio8_asr.encoder.empty()) return audio8_asr.Validate();`；`os << "audio8_asr=" ...` |
| `offline-recognizer-impl.cc` | `:39` include；`:249-251`；`:657-659` | 两处 `if (!config.model_config.audio8_asr.encoder.empty()) return std::make_unique<OfflineRecognizerAudio8AsrImpl>(config);`（放在 qwen3 之后、paraformer 之前） |
| `sherpa-onnx/csrc/CMakeLists.txt` | `:282-284`；`:835` | 加 3 个 `.cc`（model-config / model / recognizer-impl）；加 test 文件 |

### 5.5 C API / C++ API

| 文件 | 位置 | 改动 |
|---|---|---|
| `c-api/c-api.h` | `:1053` 后新增 struct；`:1129` 后加字段；`:1072` 注释列表 | `SherpaOnnxOfflineAudio8ASRModelConfig{const char* encoder; const char* decoder; const char* tokenizer; int32_t max_total_len; int32_t max_new_tokens;}`（5 字段 = 20 B）；字段**追加到 `SherpaOnnxOfflineModelConfig` 末尾** |
| `c-api/c-api.cc` | `:635-654` 之后 | 5 行 `SHERPA_ONNX_OR(...)` 映射 |
| `c-api/cxx-api.h` | `:620` 后新增 struct；`:679` 后加字段 | `std::string encoder/decoder/tokenizer; int32_t max_total_len=512; int32_t max_new_tokens=128;` |
| `c-api/cxx-api.cc` | `:383` 之后 | 5 行 `.c_str()` 赋值 |
| `wasm/nodejs/sherpa-onnx-wasm-nodejs.cc` | `:22`、`:29-53` | **必须**加 `static_assert(sizeof(SherpaOnnxOfflineAudio8ASRModelConfig) == 20, "")`，并把它的 sizeof 加进 `SherpaOnnxOfflineModelConfig` 总和断言；否则 WASM CI 失败 |

### 5.6 Python 绑定

| 文件 | 改动 |
|---|---|
| `python/csrc/offline-audio8-asr-model-config.h/.cc`（新） | copy qwen3 版，5 字段；`py::init<...>` 顺序 = C++ ctor |
| `python/csrc/offline-model-config.cc` | `:18` include；`:74` docstring；`:114` `PybindOfflineAudio8AsrModelConfig(m);`；`:135` py::init 加参；`:157` `py::arg("audio8_asr")=...`；`:180` `.def_readwrite("audio8_asr", ...)` |
| `python/csrc/CMakeLists.txt` | `:20` 后加 `.cc` |
| `python/sherpa_onnx/__init__.py` | `:31` 附近导出 `OfflineAudio8AsrModelConfig` |
| `python/sherpa_onnx/offline_recognizer.py` | `:12` import；`:478` 后新增 `from_audio8_asr(encoder, decoder, tokenizer, num_threads=1, sample_rate=16000, feature_dim=128, max_total_len=512, max_new_tokens=128, debug=False, provider="cpu")` |

### 5.7 示例

| 文件 | 改动 |
|---|---|
| `cxx-api-examples/audio8-asr-cxx-api.cc`（新） | copy `qwen3-asr-cxx-api.cc`；5 个字段 |
| `cxx-api-examples/CMakeLists.txt` | `:254` 后两行 `add_executable` / `target_link_libraries(... sherpa-onnx-cxx-api)` |
| `c-api-examples/audio8-asr-c-api.c`（新）+ `CMakeLists.txt:132` 后 | copy qwen3 版 |
| `python-api-examples/audio8-asr-decode-files.py`（新） | copy `offline-qwen3-asr-decode-files.py` |
| 可选 | `scripts/apk/...:1026`、`scripts/tauri/...:1030,1202`、`.github/scripts/flutter/generate_vad_asr_matrix.py:82` |

### 5.8 可选（首版**不建议**做）

JNI（`jni/offline-recognizer.cc:432-457`）+ `java-api/.../OfflineModelConfig.java`、Kotlin（`kotlin-api/OfflineRecognizer.kt:1000`）、Rust（`rust/sherpa-onnx/src/offline_asr.rs:325`）、Pascal（`pascal-api/sherpa_onnx.pas:420,1046`）、HarmonyOS（`harmony-os/.../non-streaming-asr.cc:160,380,500`）、Go（`go-api-examples/non-streaming-qwen3-asr-decode-files`）。这些各自镜像 C struct，**不影响 C++/Python 可用性**。

### 5.9 单测（不需要权重，务必写）

`sherpa-onnx/csrc/offline-recognizer-audio8-asr-impl-test.cc`：

1. `Audio8EncoderRows` / `Audio8AudioTokenCount`：`3000→390 / 375`、`1005→131 / 125`、`510→63 / 63`、`469→58 / 58`、`883→110 / 110`、`1764→220 / 220`、`1→1 / 1`（`K` 与 `onnx-reference.json` 的实测值逐条对齐，见 §2.2 表）。
2. `InitPromptTemplateIds` 等价性：`Encode("<|user|><|begin_of_audio|>")=={151647,151648}`、`Encode(suffix)=={151649,5501,1356,3114,419,7699,13,151650}`（**这是 §3 实测出来的黄金值**）。
3. `TrimAudioFeatures`（copy qwen3 的 3 个 case）。
4. `MaskBlockedIds`：确保 151646–151650 为 `-inf`。
5. （可选）KV 偏移单元测试：给定 `[1,8,512,64]` 与 `S=3,pos0=5`，验证写回位置。

### 5.10 与「当前正在写的实现」对照（截至 2026-11-13 19:35 的 `git status`）

另一个 agent 已在该 checkout 里落下 8 个新文件 + 6 处改动（未跟踪/未提交）。逐项对照本 checklist：

**已正确覆盖**（无需再动）：

| 项 | 证据 |
|---|---|
| 2 session（encoder + decoder，无 conv） | `offline-audio8-asr-model.cc:89-108` |
| `[B,H,L,D]` cache 布局 | `offline-audio8-asr-model.cc:179,232-233`（`kv_shape[1]`→kv_heads、`kv_shape[2]`→max_total_len） |
| `attention_mask` 定长 `max_total_len` | `offline-recognizer-audio8-asr-impl.cc:230-235, 275-279` |
| 3000 帧截断 + 零填充 + 只归一化真实帧 | 同文件 `:332-336` |
| `encoder_rows` / `K` 双计数 + 池化 | `:61-75`（`Audio8EncoderRows`、`Audio8AudioTokenCount`、`Audio8AdaptiveAvgPool`）、`:364-368` |
| `max_new_tokens` clamp | `:200` |
| stop = eos ∪ pad(151643) | `:265-267` |

**仍建议补**（按风险排序）：

1. **`Validate()` 缺 `tokenizer_config.json`**（§5.1.1）——added tokens 全靠它，缺失时**静默**退化成 BPE 碎片。
2. **没有 block ids**（151646–151650）。参考实现会 mask 掉；不 mask 时模型偶尔会吐出 `<|audio|>` 之类，而 `QwenAsrTokenizer::Decode()` 会**原样输出**（§3 实测），文本就脏了。建议在采样前把 logits 的这 5 个 id 置 `-inf`。
3. **没有退化重复保护**（Qwen3 的 `IsDegenerateRepetition`/`TrimDegenerateTail`，`impl.cc:326-389`）。Audio8 若进入 2–3 token 循环会一路跑到 `max_new_tokens`。
4. C API / cxx-api / python 绑定 / WASM `static_assert` / examples 尚未出现（§5.5–§5.7）——**不做 C API 就没有 C++ example，也无法被 python 用**。
5. 建议加断言：`audio_hidden` 的行数 == prompt 里 `<|audio|>` 的个数（现在靠同一个 `num_audio_tokens` 变量传递，正确但脆弱）。

> 说明：以上只是对照 checklist 的差异清单，不是代码评审结论；该实现仍在写入中（mtime 一直在变）。

---

## 6. 硬骨头 / open questions

### 已解决（本轮实测后不再是风险）

| # | 问题 | 结论 |
|---|---|---|
| R1 | tokenizer 是否支持 Audio8 词表/特殊 token（151643–151651） | ✅ **支持，零改动**，C++ 实跑与 HF 逐位相同（§3） |
| R2 | 是否要 host 侧 embedding（官方 bundle 311 MB） | ✅ **不需要**，导出图吃 `input_ids`，embedding + 投影都在图内 |
| R3 | audio token 数规则 | ✅ `((frames+1)//2)//4`，已用 5 条参考音频的 125/63/58/110/220 全部命中（§2.2 表） |
| R4 | mel 归一化与补零口径 | ✅ 只归一化真实帧 → 补零到 3000 帧，与 `--no-pad` 参考实现一致（`03-export.md` 的 CER=0.000 就是在这条路径上测的） |
| R5 | 是否支持 no-RoPE / 特殊位置编码 | ✅ 无关：C++ 从不碰 RoPE，全在 decoder 图内（export 侧实测为 Qwen2 结构 + RoPE theta 1e6；`config.json` 里的 `use_rope:false` 是未生效字段，不影响 C++） |
| R6 | 是否硬编码 3 图 | ✅ Qwen3 是（`offline-qwen3-asr-model.cc:135-148`）；Audio8 要 2 图，**新建模型类**而不是复用 |

### 仍阻塞/需决策

**Q1（最高）KV 缓存布局相反 → 不能照抄 `CreateEmptyKVCache` / `ApplyKvDeltaInplace`**
- 图声明 `[1,8,512,64]`（`[B,H,L,D]`），Qwen3 代码按 `[B,L,H,D]`：`offline-qwen3-asr-model.cc:311-318` 把 `tpl[1]` 当序列长度 → 会把 `max_total_len` 读成 **8**；`:543-547` 分配成 `[1,8,512→tpl[1]=8, kv_h=tpl[2]=512, hd=64]`（体积差 64 倍，直接 OOM/错误）；`:719-747` 的 `memcpy` 偏移同样错。
- 最小可行路径：**新写这两个函数**（§5.2 已给公式），不要试图在 Qwen3 里加 flag（会更难 review）。
- 验收：单测 §5.9 第 5 项 + 端到端 CER 对比。

**Q2（高）有**两个** token 计数，别混用：`num_rows`（encoder 行数）≠ `K`（prompt token 数）**
- `num_rows` 是**编码器有效行数**：`full*13 + after_cnn(rem)`，30 s → **390**，10.05 s → **131**。它同时是图内 block-diagonal 注意力的 mask 边界（`i//104==j//104 and i<num_rows and j<num_rows`，i/j 是 token 下标）。
- `K` 是 **prompt 里 `<|audio|>` 的个数**：`max(1, ((frames+1)/2)/4)`，30 s → **375**，10.05 s → **125**。
- 两者不相等，必须用 `adaptive_avg_pool1d` 把 `hidden[:num_rows]` 池化到 `K` 行再喂 decoder（官方参考实现的 `audio_tokens_from_hidden()` 就是这么做的，`export/onnx_runtime.py:98-105`）。
- 若把 `K` 直接当 `num_rows` 传（会截掉最后一个部分块的有效行）或反过来（`audio_hidden` 长度对不上 prompt，decoder 的 `gather` 会取错/越界）→ 输出乱码。
- 边缘：`frames < 8` 时 `((frames+1)/2)/4` 给 0 → `K = max(..., 1)`；`num_rows` 至少 1。更安全的是「有效帧 < 2 → 直接返回空结果」。
- guard：音频先截到 30 s，保证 `num_rows <= 390`。

**Q3（高）`attention_mask` 定长 512**
- 图要求 `[1,512]`。Qwen3 传 `[1,context_len]`（`impl.cc:946-950`）和 `[1,1]`（`:1048-1052`）**都会 shape mismatch**。
- 每步 decode 要构造/复用一块 512 长的 int64 buffer，前 `valid_len` 置 1。建议在 `GenerateText` 里预分配一次 `std::vector<int64_t> mask(mask_len, 0)`，每步只翻转一个元素再 `CreateTensor`（零拷贝用同一块内存，注意 ONNX Runtime 会不会长期持有 —— 安全做法是每步新建）。
- 从图推断 `mask_len_`（`GetInputTypeInfo(2).shape[1]`），并与 `max_total_len_` 断言相等。

**Q4（高）mel 必须定长 3000 帧 + 30 s 截断**
- sherpa 的 `OfflineStream` 不截断也不 pad；必须在自己 `Decode()` 里做（§5.3 最后一行）。
- 行为变化要写进文档/日志：**>30 s 的音频会丢尾部**（与官方 runtime 一致，但对 sherpa 用户是新行为）。
- 若要支持长音频，需要导出动态 `frames` 的 encoder（与 `num_rows` 一起），属后续增强。

**Q5（中）`max_total_len` 与 `max_new_tokens` 预算**
- `prompt_len + max_new_tokens <= 512`。30 s 时 `375+10=385`，只剩 127 → 官方默认 128 **正好越界**；官方参考实现会直接抛异常。
- 建议：`max_new_tokens = min(opt, max_seq_len - context_len)`，并在 `context_len > max_seq_len` 时打日志 + 返回空（照抄 `impl.cc:873-938` 的裁剪逻辑，但 Audio8 只能**截 audio token**，不能截 prompt 骨架）。
- 长音频场景建议导出 `max_total_len=1024` 的变体。

**Q6（中）WASM/绑定的结构体尺寸断言**
- `wasm/nodejs/sherpa-onnx-wasm-nodejs.cc:22,29-53` 有 `static_assert(sizeof(...))` 全量校验；加 C struct 必须同步，否则 WASM 构建失败。这是最容易漏的 CI 陷阱。

**Q7（中）命名歧义容易埋坑**
- `num_rows` 既不是 mel 帧数、也不是 prompt token 数，而是**编码器有效行数**（见 Q2）。图内 `allowed[i,j] = (i//104==j//104) and i<num_rows and j<num_rows`（i/j 都是 token 下标），传成 1005（帧数）会静默错。
- 建议：C++ 里分别命名 `encoder_rows` 与 `audio_token_len`，只在调用 `ForwardEncoder` 时映射成 `num_rows`。

**Q8（低）eos 集合与 block ids**
- 参考实现 stop = `{151645, 151643}`；sherpa 的 `GetEosTokenId()` 只给 151645。需自己加 151643。
- block = 151646–151650（必须），151651（`<|system|>`，建议一并 block）。

**Q9（低）hotword 语义不同**
- Qwen3：prompt 注入；Audio8：decode-time logit boosting（`start_boost=6.0`、`cont_boost=8.0`、`topk=50`，trie 前缀）。
- 首版建议：**不支持**，`Validate()` 里若有 hotwords 就 warning 并忽略（比半吊子实现安全）。

**Q10（低）文本后处理**
- `Decode()` 会原样输出特殊 token（实测），必须后过滤：截 `<|im_end|>` 之后、删 `<|...|>`、压缩空白、`.strip()`（对齐官方 `normalize_prediction_text`）。
- 中英混排要注意：官方参考输出里中文与英文单词之间有空格（`"昨天是 Monday。"`），不要额外加/去空格。

**Q11（低）模型体积与加载时间**
- encoder 839 MB + decoder 694 MB fp32；`Ort::Session` 加载在 Mac 上约数秒，`--num-threads=2` 时 RSS ≈ 1.6 GB。边缘设备需要 int8 量化变体（属导出侧工作，不在本计划内）。

---

## 7. macOS 构建计划（只编 C++ 库 + 一个 example）

### 7.1 onnxruntime：**必须下载**（无预装）

- `cmake/onnxruntime.cmake:48-79`：Darwin + arm64 + `BUILD_SHARED_LIBS=OFF` → `cmake/onnxruntime-osx-arm64-static.cmake`。
- 固定 **1.28.2**：`https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.28.2/onnxruntime-osx-arm64-static_lib-1.28.2.zip`（`SHA256=bf6e35d…38ab`，约 60 MB）。
- 离线可预放到（`onnxruntime-osx-arm64-static.cmake:14-25`）：`~/Downloads/`、`${CMAKE_SOURCE_DIR}/`、`${CMAKE_BINARY_DIR}/`、`/tmp/`。
- `brew install onnxruntime` 的库不会被自动采用（本机未装，直接走下载）。

### 7.2 实测 configure（**exit 0**）

```bash
cmake -S /tmp/sherpa-onnx -B /tmp/sherpa-survey-work/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=ON
```
- 结果：`Configuring done / Generating done`，**493 s**（几乎全耗在 ORT zip 下载+解压；纯配置 < 30 s）。
- 依赖全部 FetchContent 下载：onnxruntime 1.28.2、kaldi-native-fbank v1.23.0、simple-sentencepiece v0.7、nlohmann/json v3.12.0、kissfft、utf8proc、kaldi-decoder、eigen。
- **examples 需要三个开关同时为 ON**（`CMakeLists.txt:601-606`）：`SHERPA_ONNX_ENABLE_C_API` + `SHERPA_ONNX_ENABLE_BINARY` + `SHERPA_ONNX_BUILD_C_API_EXAMPLES`。

### 7.3 实测编译（**exit 0**）

```bash
time cmake --build build --target sherpa-onnx-core -j 10   # real 0m36.9s
time cmake --build build --target qwen3-asr-cxx-api -j 10  # real 0m4.9s
```

产物（实测）：

```
build/bin/qwen3-asr-cxx-api        31,781,216 B   ← 静态链接 ORT，可直接跑
build/lib/libsherpa-onnx-core.a    11,313,856 B
build/lib/libsherpa-onnx-c-api.a      161,160 B
build/lib/libsherpa-onnx-cxx-api.a    156,168 B
build/lib/lib{kaldi-native-fbank-core,kaldifst-core,kaldi-decoder-core,
             fst,fstfar,ssentencepiece_core,kissfft-float,utf8proc}.a
```

唯一告警：Xcode CLT 的 `The selected platform is no longer supported by libc++`（来自 `CMAKE_OSX_DEPLOYMENT_TARGET=10.14`，`CMakeLists.txt:23`），不影响产物。

> 时间结论：**从零 ≈ 8 min 配置（含 60 MB 下载）+ 40 s 编译**；example 只额外编 3 个 TU，改代码后增量重编 **5–15 s**，迭代成本很低。

### 7.4 实现者的推荐命令

```bash
cmake -S /tmp/sherpa-onnx -B build-audio8 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_TTS=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=ON

cmake --build build-audio8 --target audio8-asr-cxx-api -j 10
./build-audio8/bin/audio8-asr-cxx-api $WAV \
  --audio8-asr-encoder=audio8/exported/encoder.onnx \
  --audio8-asr-decoder=audio8/exported/decoder.onnx \
  --audio8-asr-tokenizer=audio8/downloaded/hf_model
```

（本机已有 `audio8/exported/{encoder,decoder}.onnx` 与 `audio8/downloaded/hf_model/`，可直接用来跑验收。）

---

## 8. 验收口径

1. **tokenizer**：§5.9 的黄金 id 断言全绿。
2. **特征**：同一 wav，`OfflineStream(WhisperTag{128,true})` 的帧数 = `floor(n/160)`，与 HF `--no-pad` 参考的 `mel_frames` 一致（例：10.05 s → 1005）。
3. **两个 token 计数**：`K` 与 `onnx-reference.json` 的 `audio_tokens` 逐条相同（125 / 63 / 58 / 110 / 220）；`encoder_rows` 按 `full*13+after_cnn(rem)` 独立算（30 s → 390），并断言 `pool(encoder_rows → K)` 不抛异常。
4. **端到端文本**：5 条测试 wav 的输出与 `hf-reference-unpadded.json` **逐字一致**（CER=0）—— 这是最硬的验收，导出侧已做到，C++ 侧应能复现。
5. **回归**：`qwen3-asr-cxx-api` 仍能跑通；`sherpa-onnx-core` 无新增告警。
6. **边界**：全静音 wav → 空串；35 s wav → 截断警告且不崩；<0.16 s wav → 不越界。
7. **多线程**：`sherpa-onnx-offline-parallel` 风格并发（同一 recognizer 多线程 `DecodeStreams`）无数据竞争（`rng_mutex_` 模式照抄）。

---

## 9. 工作量粗估（对照 §5 的 checklist）

| 阶段 | 估计 |
|---|---|
| config 补齐（§5.1.1 的 3 处） | 0.5 h |
| 模型类（2 session + KV `[B,H,L,D]` + 定长 mask 推断） | 4–6 h |
| recognizer impl（mel pad/截断 + num_rows + prompt + block ids + stop 集合 + 后处理） | 6–8 h |
| 注册 / C API / C++ API / python 绑定 | 3–4 h |
| example ×2 + python example | 1–2 h |
| 单测（§5.9 五项） | 2 h |
| 特征/文本数值对齐 + 边界 | 3–4 h |
| **合计** | **≈ 2.5–3 人日** |

---

## 附录 A：未做/未验证

- 未验证 C++ 侧端到端（没有编 `OfflineAudio8AsrModel`；本报告只做调研与计划）。
- 未跑 `ctest`（本机未开 `SHERPA_ONNX_ENABLE_TESTS`）；WASM `static_assert` 风险来自**读代码**（`wasm/nodejs/sherpa-onnx-wasm-nodejs.cc:22,29-53`），未实编。
- 未量化 int8 变体的精度/体积（属导出侧）。
- 未检查 `audio8/exported/*.onnx` 是否可被 sherpa 依赖的 **ONNX Runtime 1.28.2** 加载（导出用的是 ORT 1.23.2；opset 过高会加载失败 —— 实现第一步就该用 §7.4 的构建产物试加载一次）。
- `/tmp/sherpa-onnx` 全程只读（两个新文件由另一 agent 写入，非本会话）；构建与实验产物在 `/tmp/sherpa-survey-work/`（`build/`、`tokcheck/`、`whisper-feature.h`、`configure.log`、`build-core.log`、`build-example.log`）。
