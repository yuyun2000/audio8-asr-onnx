# Audio8-ASR-0.1B → ONNX + sherpa-onnx

把 [`Edge0/Audio8-ASR-0.1B`](https://huggingface.co/Edge0/Audio8-ASR-0.1B)（Whisper 风格音频编码器 + 0.1B Qwen2 解码器）
转成 ONNX，并在 sherpa-onnx 里写上 C++ runtime（PR: <https://github.com/k2-fsa/sherpa-onnx/pull/4198>）。

## 目录

| 路径 | 内容 |
|---|---|
| `export/export_audio8_onnx.py` | 导出 `encoder.onnx` + `decoder.onnx`（静态 shape） |
| `export/onnx_runtime.py` | 纯 ONNX 参考实现（词表、prompt、贪心解码） |
| `export/hf_reference.py` | transformers 参考实现（含 transformers 5.22 两个 bug 的修复） |
| `reports/01-model-spec.md` | 模型规格（结构、prompt、token 公式、张量契约） |
| `reports/03-export.md` | 导出过程与精度验证 |
| `reports/04-cpp-runtime.md` | C++ 集成、踩坑与实机数字 |
| `ref/` | 官方 ONNX runtime 对照（transcripts、parity、张量 dump） |

| 资源 | 地址 |
|---|---|
| 上游 PR | <https://github.com/k2-fsa/sherpa-onnx/pull/4198> |
| ONNX 模型（encoder 839 MB + decoder 694 MB + tokenizer） | <https://huggingface.co/yunyu1258/audio8-asr-onnx> |
| 模型权重 | <https://huggingface.co/Edge0/Audio8-ASR-0.1B> |
| 官方 ONNX runtime 参考实现 | <https://huggingface.co/Edge0/Audio8-ASR-0.1B-onnx-runtime> |

## 快速开始

```bash
export HF_HOME=$PWD/downloaded/hf_home
huggingface-cli download Edge0/Audio8-ASR-0.1B --local-dir downloaded/hf   # 需 HF_ENDPOINT 镜像
python export/export_audio8_onnx.py --out-dir exported
python export/onnx_runtime.py wavs/0.wav
```

编译 sherpa-onnx 并跑 C++：

```bash
cd ../sherpa-onnx   # 含本 PR 分支 audio8-asr
cmake -B build -DSHERPA_ONNX_ENABLE_BINARY=ON -DSHERPA_ONNX_ENABLE_TESTS=OFF ...
cmake --build build --target sherpa-onnx-offline -j
DYLD_FALLBACK_LIBRARY_PATH=/usr/lib ./build/bin/sherpa-onnx-offline \
  --audio8-asr-encoder=../audio8/exported/encoder.onnx \
  --audio8-asr-decoder=../audio8/exported/decoder.onnx \
  --audio8-asr-tokenizer=../audio8/downloaded/hf wavs/0.wav
```

## 关键事实

- 特征：Whisper log-mel，128 bins，25 ms/10 ms，补零到 3000 帧（30 s）
- 音频编码器：3×conv(stride2) + 18 层 attention（d=896，14 头，块对角窗口 104 token）+ 4 层 MLP tower
- 音频 token：`K = ((mel_frames+1)/2)/4`；编码器行数 `E = 13*(frames//100)+after_cnn(frames%100)`，两者用 adaptive-avg-pool 桥接
- 解码器：Qwen2 8 层、hidden 512、8 头×64、RoPE θ=1e6、tie embeddings、vocab 151936
- prompt：`151647 151648 [151646]×K 151649 "Please transcribe this audio." 151650`（无 system prompt）
- 精度：5 条音频 C++ 与 transformers 逐字一致（CER 0.000）；RTF 0.075–0.204（Mac CPU）
