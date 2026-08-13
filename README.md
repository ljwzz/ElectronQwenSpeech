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

## 迁移来源

`mps` 固定迁移自 `/Users/paas/code/voxweaver-tts` 提交
`d865b8a6e2c96d5e5a1d786758f4b61124bc5720`。迁移不读取该提交之后的
MLX 工作树改动，也不修改源仓状态。

`mlx` 固定迁移自同一源仓的实现提交
`b86f9ce1adf6a6a8863efd2067e4c791c386d4f6` 与验证报告提交
`04c7a61313f9b4e273e90848009aff579b08b994`。两个运行时分支互不继承实现提交。
