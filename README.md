# ElectronQwenSpeech

ElectronQwenSpeech 是独立的 Qwen 本地语音诊断项目。仓库使用运行时分支隔离不同实现，`main` 只维护分支说明，不承载可运行代码。

## 分支

| 分支 | 状态 | 说明 |
| --- | --- | --- |
| `main` | 说明分支 | 维护分支用途与迁移来源。 |
| `mps` | 可运行实现 | PyTorch/MPS 语音诊断实现，使用 `mps:0`、`bfloat16`，禁止 CPU fallback。 |
| `mlx` | 可运行实现 | MLX/Metal 语音诊断实现，使用 `gpu:0` 与固定本地 BF16 模型。 |

切换到对应实现：

```bash
git switch mps
git switch mlx
```
