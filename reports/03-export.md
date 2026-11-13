# Audio8-ASR-0.1B → ONNX 导出与精度验证（本机 macOS，无 AX 卡）

日期：2026-11-13 · 环境：macOS / Python 3.12（torch 2.15、transformers 5.22.2、onnxruntime 1.23.2，纯 CPU）

## 结论

| 项 | 结果 |
|---|---|
| 导出的图 | `encoder.onnx`（839 MB）+ `decoder.onnx`（694 MB），fp32 |
| 精度（5 条测试音频） | ONNX 与 transformers 参考**逐字一致（CER = 0.000）** |
| 推理耗时（CPU） | 0.9–1.4 s / 条（10–17 s 音频） |
| 显存/内存 | 约 1.6 GB RSS（fp32 权重常驻） |

## 模型结构（实测）

| 部分 | 规格 |
|---|---|
| 特征 | Whisper 风格 log-mel，128 bins，n_fft 400，hop 160，补零到 3000 帧（30 s） |
| 音频编码器 | conv2d×3（stride 2）+ 18 层 self-attention（d=896、14 头、ffn 3584），块对角注意窗口 104 token（`n_window_infer // (n_window*2) = 8` 个 100 帧块） |
| 音频 token | 每 100 mel 帧 13 个 token；`N = ((mel_frames+1)//2)//4` |
| MLP tower | 4 层 LayerNorm+Linear(1024)+GELU+Linear，作用在编码器输出 |
| 投影 | LayerNorm + Linear(1024 → 512) |
| LM | Qwen2 结构，8 层、hidden 512、8 头×64、RoPE（theta 1e6）、SwiGLU、tie_word_embeddings |
| 词表 | 151936（embedding 与 lm_head 共享，HF 发布版删了 lm_head，必须手动 tie） |
| Prompt | `<|user|><|begin_of_audio|>` + `<|audio|>`×N + `<|end_of_audio|>` + `Please transcribe this audio.` + `<|assistant|>`（与官方 ONNX runtime 一致；**没有 system prompt**） |
| 结束符 | 151645（`<|im_end|>`），pad 151643 |

## 导出设计（两个图）

### encoder.onnx

```
输入: audios    float32 [1, 128, 3000]       # Whisper log-mel，零填充
      num_rows  int64   [1]                  # 真实音频长度对应的行数
输出: hidden    float32 [1, 390, 1024]       # MLP tower 输出，第 num_rows 行之后无效
```

HF 的编码器用了 `pad_sequence` 和 `split(chunk_lengths.tolist())`，两者都带数据依赖，无法直接 trace。导出里把 100 帧分块写死成 `view(30, 100, 128)`（块的划分与原实现完全一致），注意力用「104 token 一块 + `num_rows` 行掩码」的常量 mask 复现变长行为：

```
allowed[i, j] = (i // 104 == j // 104) and i < num_rows and j < num_rows
```

这样最后一个不满 100 帧的块里，只有真实存在的行参与注意力，与 HF 的 `cu_seqlens` 行为逐位一致。（第一版没加 `num_rows`，末尾零行混进注意力窗口，导致 5 条里 3 条有 1–2 字漂移，CER 最高 0.286。）

### decoder.onnx

```
输入: input_ids       int64   [1, T]
      audio_hidden    float32 [1, K, 1024]   # 编码器输出按真实行数池化到 K 个音频 token
      attention_mask  int64   [1, 512]
      cache_position  int64   [T]
      cache_key_i / cache_value_i  float32 [1, 8, 512, 64]   (i = 0..7)
输出: logits          float32 [1, T, 151936]
      present_key_i / present_value_i  float32 [1, 8, T, 64]
```

实现要点：

- 音频注入用 `cumsum(mask) - 1` + `gather` + `where`，不用 `Scatter`（ONNX ScatterElements 在动态 shape 下越界）；
- KV cache 固定 512 槽，当前 token 的 K/V 通过 one-hot 矩阵加进缓存槽位，输出「本次新增」的 K/V 切片，由运行时按 `cache_position` 写回；
- 因果 mask 用 `cache_position` 与槽位下标比较生成，decode 时 `T=1`；
- 投影层（LayerNorm + Linear 1024→512）放在 decoder 图里，编码器只输出 1024 维 hidden。

## 精度验证

参考实现：transformers `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`，fp32、eager attention、greedy、`max_new_tokens=128`，处理器给的是**未补零**的 mel（1005 帧等）。

| wav | 音频 | transformers 参考 | ONNX 导出 | CER |
|---|---|---|---|---|
| 0.wav | 10.05 s | 昨天是 Monday。Today is礼拜二。The day after tomorrow日是星期三。 | 同左 | **0.000** |
| 1.wav | 5.10 s | 这是第一种。第二种叫呃，与always，os什么意思啊？ | 同左 | **0.000** |
| 2.wav | 4.69 s | 就是平凡的，就是静下来，复语款的，平凡的。 | 同左 | **0.000** |
| 3.wav | 8.83 s | 第一句是个什么十台加了，夜四时，一百现代式，这个什么海市。 | 同左 | **0.000** |
| 4.wav | 17.64 s | 嗯，on time就要准时，…他现在正在教他的。 | 同左 | **0.000** |

逐张量对比：

- encoder：ONNX vs torch 编码器 cosine ≈ 1.0（max|diff| ≈ 1e-4）
- decoder prefill logits：cosine ≈ 1.0（max|diff| ≈ 4e-5），argmax 一致

## 复现

```bash
cd audio8/export
# 1) 参考文本（transformers）
python hf_reference.py --no-pad ../wavs/*.wav
# 2) 导出 ONNX
python export_audio8_onnx.py --out-dir ../exported
# 3) ONNX 推理 + 对比
python onnx_runtime.py ../wavs/*.wav
```
