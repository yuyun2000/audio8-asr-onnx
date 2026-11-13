# Qwen3-ASR 离线识别调用链（sherpa-onnx）

> 全部 `file:line` 基于 `/tmp/sherpa-onnx` HEAD `031a93a`。
> 记号：`→` 调用/数据流，`⟂` 提前返回，`[n]` 第 n 步。

```text
┌─────────────────────────── 创建阶段（一次） ───────────────────────────┐
│                                                                        │
│  用户配置                                                              │
│    OfflineRecognizerConfig                                             │
│      ├─ model_config.qwen3_asr = OfflineQwen3ASRModelConfig{...}       │
│      │     (offline-qwen3-asr-model-config.h:14-69)                    │
│      └─ feat_config  ← 本路径**不使用**（特征器由 CreateStream 决定）  │
│                                                                        │
│  [1] OfflineRecognizer::OfflineRecognizer(config)                      │
│        offline-recognizer.cc:159-160                                   │
│        └→ OfflineRecognizerImpl::Create(config)                        │
│             offline-recognizer-impl.cc:249-251                         │
│             if (!config.model_config.qwen3_asr.conv_frontend.empty())  │
│                 → new OfflineRecognizerQwen3ASRImpl(config)            │
│             （Android asset / OHOS 走 :657-659 的 manager 重载）        │
│                                                                        │
│  [2] OfflineRecognizerQwen3ASRImpl ctor                                │
│        offline-recognizer-qwen3-asr-impl.cc:405-414                    │
│        ├─ OfflineRecognizerImpl(config)        // ITN / 同音替换器     │
│        ├─ model_  = OfflineQwen3ASRModel(model_config)                 │
│        │     offline-qwen3-asr-model.cc:799-800 → Impl():125-151       │
│        │       ├─ 3 个 Ort::Session                                    │
│        │       │    conv_frontend :135-138 (1 进 / 1 出, :229-239)     │
│        │       │    encoder       :140-143 (2 进 / 1 出, :261-272)     │
│        │       │    decoder       :145-148 (4+2L 进 / 1+2L 出,         │
│        │       │                            :293-309)                  │
│        │       └─ InitDecoderSession() :311-343                        │
│        │            max_total_len_ ← past_key 输入静态 shape[1]        │
│        │            kv 元素类型 ∈ {f32, f16, u16}                      │
│        ├─ tokenizer_ = QwenAsrTokenizer("<dir>")                       │
│        │     qwen-asr-tokenizer.cc:1161-1197                           │
│        │       vocab.json → merges.txt → tokenizer_config.json         │
│        │       (ParseAddedTokens 读 added_tokens_decoder :838-899)     │
│        │       eos = <|im_end|>  (:1150-1156)                          │
│        └─ InitPromptTemplateIds()   :439-457                           │
│              audio_pad_ids_   = Encode("<|audio_pad|>")                │
│              prompt_ids_after_= Encode("<|audio_end|><|im_end|>\n"     │
│                                       "<|im_start|>assistant\n")       │
│              asr_text_token_id_ = GetTokenId("<asr_text>")  // 缺失即死│
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────── 每次识别（可多流/多线程） ────────────────────┐
│                                                                        │
│  [3] stream = recognizer.CreateStream()                                │
│        OfflineRecognizer::CreateStream()  offline-recognizer.cc:169-171│
│        └→ impl->CreateStream()            impl.cc:428-437              │
│             WhisperTag{ dim = 128,                 // kQwen3MelDim :47 │
│                         align_to_stft_center = true }  // :435         │
│             → new OfflineStream(tag)                                   │
│                 offline-stream.cc:82-107                               │
│                   knf::OnlineWhisperFbank(16 kHz, 25 ms, 10 ms, hann,  │
│                                           snip_edges=false, 128 mel)   │
│                   whisper_center_padding_ = 80 采样点（半帧移反射填充）│
│                                                                        │
│  [4] stream.AcceptWaveform(16000, samples, n)                          │
│        offline-stream.cc:200-215                                       │
│        └→ 首块先补 80 点反射 padding → whisper_fbank_->AcceptWaveform  │
│                                                                        │
│  [5] recognizer.Decode(&stream)  → DecodeStreams(&s,1)                 │
│        offline-recognizer.cc:173-181 → impl.cc:1149-1154 → Decode()    │
│                                                                        │
│  [6] Decode(stream)                       impl.cc:1156-1236            │
│        f = stream->GetFrames()            // 1-D, num_frames*128      │
│        │     offline-stream.cc:232-260                                 │
│        │       n = min(fbank 帧数, wav_len/160)  ← 去掉补 padding 的尾帧│
│        ├─ 空 / 不足 2 帧 ⟂ 空结果                                       │
│        ├─ NormalizeWhisperFeatures(f, T, 128)     impl.cc:1184         │
│        │     math.cc:115-135: log10(clamp(x,1e-10)) → max(·,max-8)     │
│        │                        → (x+4)/4                             │
│        ├─ conv_input = Tensor[1, T, 128] f32      :1189-1194           │
│        ├─ [7] ForwardConvFrontend(conv_input)     :1196                │
│        │        model.cc:347-372 → [1, T', C']                         │
│        ├─ expected = FeatToAudioTokensLen(T, 100) :1207-1208           │
│        │     impl.cc:77-108  (3× stride-2 conv ⇒ 100 帧 → 13 token)    │
│        ├─ mask[1, T'] bool：前 min(expected,T') 为 true  :1210-1218    │
│        ├─ [8] ForwardEncoder(conv_out, mask)      :1220-1221           │
│        │        model.cc:374-413 → audio_features [1, A, H]            │
│        └→ [9] GenerateText(audio_features, valid_frames, stream)       │
│                                                                        │
│  [9] GenerateText(...)                    impl.cc:765-1147             │
│        ├─ 读 per-stream 选项: max_new_tokens / temperature / top_p     │
│        │     (stream->GetOptionInt/Float, offline-stream.h)            │
│        ├─ TrimAudioFeatures(audio_features)  :784-785 → :129-207       │
│        │     尾部全静音帧裁掉；全静音时 all_silent=true                │
│        ├─ audio_token_len = min(audio_token_len, A)  :789-792          │
│        ├─ all_silent ⟂ 空结果（不再构造 prompt）  :794-799              │
│        ├─ hotwords = Qwen3FormatHotwordsForPrompt(csv)  :819-821       │
│        ├─ source_ids = BuildSourceIds(...)       :830-831 → :459-507   │
│        │     "<|im_start|>system\n" + hotwords + "<|im_end|>\n"        │
│        │     "<|im_start|>user\n<|audio_start|>"                       │
│        │     + audio_pad_ids_ × audio_token_len                        │
│        │     + (可选 "language X" + <asr_text>)                        │
│        │     + "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"      │
│        ├─ cache_kv = CreateEmptyKVCache(1)   :839-840                  │
│        │     model.cc:533-581 → L 对 [1, max_total_len, kv_h, hd] 全 0 │
│        ├─ max_seq_len = min(model_max_len, opt)  :841-847              │
│        ├─ 超长时：截 audio_pad 个数 + TruncateAudioFeatures :873-938   │
│        ├─ ── prefill ──                                                │
│        │   input_ids[1,ctx] i64 / attn_mask[1,ctx] 全 1 /              │
│        │   cache_position = [0..ctx-1]  :940-953                       │
│        │   [10] model_->ForwardLLM(...)   :956-958                     │
│        │        model.cc:415-531 → logits[1,ctx,V] + L×(k,v) delta     │
│        │   [11] ApplyKvDeltaInplace(cache, delta, cache_position)      │
│        │        :962 → model.cc:583-749                                │
│        │          pos0 = cache_position[0]; memcpy 到                   │
│        │          (b*max_total_len + pos0)*(kv_h*hd*elem_bytes)        │
│        ├─ first = SampleTokenFromLogits(logits, ctx-1, T, top_p)       │
│        │     :988 → :544-584 → :586-763                                │
│        │       temperature<=1e-6 ⇒ argmax；否则 softmax+top-p+mt19937  │
│        ├─ first == eos ⇒ 用 avoid_id=eos 重采，仍 eos ⟂ 空  :990-1023  │
│        └─ ── decode loop: step 1..max_new_tokens-1 ──   :1028-1103     │
│             每步：                                                      │
│               input_ids=[1,1]、attn_mask=[1,1]、                       │
│               cache_position=[cur_len]                  :1041-1058     │
│               ForwardLLM → logits[1,1,V]                :1062-1064     │
│               ApplyKvDeltaInplace                       :1068          │
│               next = SampleTokenFromLogits(logits, 0)   :1080          │
│               next == eos ⇒ break                       :1082-1084     │
│               IsDegenerateRepetition(末尾64内≤3种id) ⇒                  │
│                   打日志 + TrimDegenerateTail + break    :1089-1102    │
│              guard: cur_len >= max_seq_len ⇒ break      :1029-1031     │
│              guard: step+1==max_new_tokens ⇒ 打截断警告 :1033-1039     │
│                                                                        │
│  [12] 结果组装（仍在 GenerateText 内）  :1105-1146                     │
│        ├─ 若前 16 token 出现 <asr_text> 且前缀解码为                    │
│        │   "language X<asr_text>" ⇒ 剥掉前缀             :1105-1123    │
│        ├─ result.text = tokenizer_->Decode(ids)          :1125         │
│        │     qwen-asr-tokenizer.cc:1255-1284                           │
│        │       <|im_start|>/<|im_end|> 跳过，其余特殊 token 原样输出    │
│        ├─ RemoveUtf8ReplacementChars(&text)              :1126         │
│        └─ result.tokens = 逐 id GetTokenStringStreaming() :1128-1144   │
│                                                                        │
│  [13] 回到 Decode():  r.text = ApplyHomophoneReplacer(text)  :1233     │
│                       stream->SetResult(r)                  :1235     │
│                                                                        │
│  [14] OfflineRecognizer::DecodeStreams: 对每个流                       │
│        r.text = RemoveLeadingSpaces(r.text); SetResult    :176-180     │
│                                                                        │
│  [15] user: recognizer.GetResult(&stream) → OfflineRecognitionResult   │
│        可用字段：text、tokens（无时间戳/置信度）                       │
└────────────────────────────────────────────────────────────────────────┘
```

## 关键点速查

| 问题 | 答案 | 证据 |
|---|---|---|
| 谁决定 mel 参数？ | `CreateStream()` 里的 `WhisperTag`，不是 `feat_config` | `impl.cc:428-437` |
| mel 是 128 bin 吗？ | 是，`kQwen3MelDim=128` | `impl.cc:47` |
| n_fft / hop | 400 / 160（25 ms / 10 ms @16 kHz） | `cmake` 依赖 knf v1.23.0 `WhisperFeatureOptions` |
| 帧对齐 | 中心化 STFT（补半个帧移，去掉尾帧） | `offline-stream.cc:93-104, 239-250` |
| 归一化 | Whisper 式，全局 | `math.cc:115-135` |
| 送模型的 mel 形状 | `[1, T, 128]`，T 动态，不 pad | `impl.cc:1189-1194` |
| audio token 数从哪来 | 解析式，不读图 | `impl.cc:77-108` |
| KV 布局 | `[B, L, H, D]`（序列在第 1 维） | `model.cc:543-547, 719-747` |
| cache 容量 | 图输入静态 shape 优先 | `model.cc:311-318` |
| 默认采样 | greedy（`temperature=1e-6`） | `offline-qwen3-asr-model-config.h:37` |
| 停止条件 | 仅 `<|im_end|>` | `impl.cc:967, 1082` |
| 结果字段 | `text` + `tokens` | `impl.cc:1125-1143` |

---

## 附：Audio8-ASR 相对本流程的差异（只列替换点）

本图是 **Qwen3-ASR** 的路径。Audio8 复用同一骨架，需替换以下 6 处（详见 `02-sherpa-integration-plan.md` §5.2/§5.3）：

| 流程节点 | Qwen3-ASR | Audio8-ASR |
|---|---|---|
| [2] 图数量 | 3 个 session（conv / encoder / decoder） | **2 个**（encoder / decoder），没有 conv_frontend |
| [3][4][5] 特征 | 同上（`WhisperTag{128, center}`）**完全复用** | 同上，无改动 |
| [6] 特征 → encoder | `NormalizeWhisperFeatures` → `[1,T,128]` → conv → encoder(conv_out, bool mask) | `NormalizeWhisperFeatures` → **截到/补零到 `[1,128,3000]`** → `encoder_rows=full*13+after_cnn(rem)` → `encoder(audios, encoder_rows)` → `hidden[1,390,1024]` → **`hidden[:encoder_rows]` 池化到 `K=((frames+1)/2)/4`**（两个计数不等，必须池化） |
| [9] prompt | `<|im_start|>system…<|audio_start|>` + `<|audio_pad|>`×A + `<|audio_end|>…<asr_text>` | `<|user|><|begin_of_audio|>` + `<|audio|>`×K + `<|end_of_audio|>Please transcribe this audio.<|assistant|>`（**K 个单 id，无 asr_text**） |
| [9] prefill / decode mask | `attention_mask[1,context_len]` 全 1 / 每步 `[1,1]` | **定长 `[1,512]`**：prefill 前 context_len 置 1；每步 decode 前 `cur_len+1` 置 1 |
| [11] KV 写回 | `[B, L, H, D]`，`max_total_len` 取 `tpl[1]` | **`[B, H, L, D]`**，`max_total_len` 取 `tpl[2]`；delta 写回偏移按 `(b*kv_h + h)*max_total_len + pos0` |
| [12] 停止/清理 | eos=`<|im_end|>`；剥 `language X<asr_text>` | stop=`{151645, 151643}`；**block 151646–151650**；过滤 `<|...|>` 字面串 |
