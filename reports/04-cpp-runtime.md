# Audio8-ASR-0.1B 的 C++ runtime（sherpa-onnx 集成）

日期：2026-11-13 · 环境：macOS（Apple M 系列，CPU fp32）· PR：<https://github.com/k2-fsa/sherpa-onnx/pull/4198>

## 结论

| 项 | 结果 |
|---|---|
| 集成方式 | 新增模型类型 `audio8-asr`（config + model + recognizer impl），复用已有的 Whisper 特征提取、`QwenAsrTokenizer` |
| 编译 | macOS 上 `cmake --build --target sherpa-onnx-offline` 通过（onnxruntime 1.28.2 osx-arm64 静态库） |
| 精度 | 5 条测试音频，C++ 输出与 transformers（修正后）**逐字一致（CER 0.000）** |
| 速度 | RTF 0.075–0.204（10 s 音频约 1.0–1.1 s） |

## 新增/修改的文件

```
sherpa-onnx/csrc/offline-audio8-asr-model-config.{h,cc}     # 配置 + 校验
sherpa-onnx/csrc/offline-audio8-asr-model.{h,cc}            # 两个 ONNX session + KV cache 管理
sherpa-onnx/csrc/offline-recognizer-audio8-asr-impl.{h,cc}  # 特征、prompt、池化、贪心解码
sherpa-onnx/python/csrc/offline-audio8-asr-model-config.*   # Python 绑定
sherpa-onnx/python/sherpa_onnx/offline_recognizer.py        # from_audio8_asr 工厂
python-api-examples/offline-audio8-asr-decode-files.py
scripts/audio8-asr/{export-onnx.py,run-onnx.py,README.md}   # 转换脚本与说明
+ 6 处注册/CMake 改动
```

用法：

```bash
./build/bin/sherpa-onnx-offline \
  --audio8-asr-encoder=./exported/encoder.onnx \
  --audio8-asr-decoder=./exported/decoder.onnx \
  --audio8-asr-tokenizer=./Audio8-ASR-0.1B \
  --num-threads=4 wavs/0.wav
```

## 数据流

```
wav --(sherpa WhisperTag, 128 mel)--> frames
    --NormalizeWhisperFeatures--> [T,128] --零填充--> [1,128,3000]
    --encoder.onnx(audios, num_rows)--> hidden [1,390,1024]
    --取前 num_rows 行 + adaptive_avg_pool--> [K,1024]
    --prompt: 151647 151648 [151646]*K 151649 "Please transcribe this audio." 151650
    --decoder.onnx prefill(input_ids, audio_hidden, mask[1,512], pos[0..T-1], 空 cache)
    --贪心解码--> 每步 decoder.onnx(pos=[p], mask 前 p+1 为 1) + KV delta 写回
    --QwenAsrTokenizer::Decode--> 文本（过滤 block id、重复退化保护）
```

## 踩到的坑（都已在代码里处理）

| 坑 | 现象 | 处理 |
|---|---|---|
| KV cache 布局 | 图是 `[1, 8, 512, 64]`（`[B,H,L,D]`），Qwen3-ASR 的实现按 `[B,L,H,D]` 写 | 按 head 步长 `max_total_len*head_dim` 拷贝，每个 head 内连续拷贝 `seq_len*head_dim` |
| 未初始化的 cache | 首次跑出全 `!`，第二次又正常（allocator 残留） | `CreateEmptyKVCache` 里显式清零 |
| `attention_mask` 定长 | 图要求 `[1,512]`，不能传 `[1,T]` | 固定 512，前 p+1 置 1 |
| 两种 token 计数 | prompt 里 `<|audio|>` 数 `K=((frames+1)/2)/4` ≠ 编码器行数 `E=13*(frames//100)+after_cnn(frames%100)` | 先取前 E 行，再 adaptive-avg-pool 到 K |
| 30 s 上限 | prompt 385 + 128 > 512 cache | `max_new_tokens` 自动截到 `max_total_len - prompt_len - 1` |
| transformers 5.22 两个静默 bug | `lm_head` 未 tie（乱码）；正弦位置表被清零（输出退化） | 导出脚本里显式修复（见 `scripts/audio8-asr/README.md`） |

## 精度验证（C++ 与参考逐字一致）

| wav | 时长 | 输出 |
|---|---|---|
| 0 | 10.05 s | 昨天是 Monday。Today is礼拜二。The day after tomorrow，是星期三。 |
| 1 | 5.10 s | 这是第一种。第二种叫呃，与os、os什么意思啊？ |
| 2 | 4.69 s | 就是平凡的啊，不认识接下来， frequent平凡的。 |
| 3 | 8.83 s | 第一句是个什么时代？加了ES是一般现代式，对，我们把它时代一些商品。 |
| 4 | 17.64 s | 嗯，On time 就准时，in time 是及时交。他总是准时交他的作业。… |

耗时（4 线程）：1.06 / 0.98 / 0.96 / 1.05 / 1.32 s，RTF 0.075–0.204。

## 本机编译方法（macOS）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_BINARY=ON
cmake --build build --target sherpa-onnx-offline -j
# 运行（系统 libc++ 不在 rpath 里）：
DYLD_FALLBACK_LIBRARY_PATH=/usr/lib ./build/bin/sherpa-onnx-offline ...
```

## 后续

- AX 平台：模型是纯 ONNX（静态 shape：30 s 窗口、512 cache 槽），到公司后可走 Pulsar2 转 AXMODEL；encoder 是 CNN+attention（14 头），decoder 是 8 层 Qwen2，都有成熟量化方案。
- 长音频：需要 VAD 切分（模型只有 30 s 窗口）。
