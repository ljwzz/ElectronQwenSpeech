# ElectronQwenSpeech

ElectronQwenSpeech 是独立的 Qwen 本地语音诊断项目。仓库以运行时分支隔离实现，
`main` 只维护分支与迁移说明，不承载可运行代码。

## 分支

| 分支 | 状态 | 说明 |
| --- | --- | --- |
| `main` | 说明分支 | 维护分支用途与迁移来源。 |
| `mps` | 独立实现 | PyTorch/MPS 语音诊断实现。 |
| `mlx` | 当前实现 | MLX/Metal 语音诊断实现。 |

切换到 MLX 实现：

```bash
git switch mlx
```

## MLX 实现边界

`mlx` 只提供 MLX/Metal `gpu:0` 运行路径，不提供运行时选择器、PyTorch 依赖或
自动 fallback。主模型权重固定为 BF16，TTS speech tokenizer 保持 FP32。模型文件
是本机前置条件，不纳入仓库。

模型 revision 固定，路径不设默认值。首次运行时创建本机配置：

```bash
cp .env.example .env.local
```

然后将 `.env.local` 中的三项值改为本机存在的绝对模型目录。`.env.local`
已忽略追踪，`.env.example` 只保留变量名和占位路径。

| 环境变量 | 固定 revision |
| --- | --- |
| `ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH` | `52f4770fd9726457eae3d3b6aa92047a25a10776` |
| `ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH` | `e1f6c266914abc5a46e8756e02580f834a6cf8a7` |
| `ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH` | `53c8c0e46733eec430e4b53dd6471d0e5dee45f8` |

根脚本 `dev`、真实模型测试、完整性校验和 benchmark 会自动加载 `.env.local`。
如果父进程已设置同名变量，父进程值优先；不使用 `VITE_` 前缀，路径不会暴露给
Renderer。

依赖固定为 `mlx-audio==0.4.8`、`mlx==0.32.0` 与 `mlx-lm==0.31.3`。
MLX 官方安装要求见 <https://ml-explore.github.io/mlx/build/html/install.html>；
MLX-Audio v0.4.8 的 Qwen3-TTS 精度修复见
<https://github.com/Blaizzy/mlx-audio/releases/tag/v0.4.8>。

## 环境与安装

- Node.js `24.13.0`；
- pnpm `11.5.2`；
- Python `3.12`。

安装 Node.js workspace 依赖：

```bash
fnm use
corepack pnpm install
```

ASR/Forced Aligner 与 TTS 使用两个隔离的 MLX 环境，每个 Sidecar 只有一份
`requirements.txt`：

```bash
python3.12 -m venv services/asr-sidecar/.venv-mlx
services/asr-sidecar/.venv-mlx/bin/python -m pip install -r services/asr-sidecar/requirements.txt

python3.12 -m venv services/tts-sidecar/.venv-mlx
services/tts-sidecar/.venv-mlx/bin/python -m pip install -r services/tts-sidecar/requirements.txt
```

## 开发与验证

启动唯一的 MLX 诊断应用：

```bash
corepack pnpm run dev
```

运行普通检查、脚本测试和固定模型完整性校验：

```bash
corepack pnpm run check
corepack pnpm run test:scripts
corepack pnpm run validate:speech-models:mlx
```

真实模型测试默认且仅使用 MLX：

```bash
corepack pnpm run test:asr:real
corepack pnpm run test:tts:real
corepack pnpm run test:speech:real
```

运行单运行时 benchmark：

```bash
corepack pnpm run benchmark:tts:mlx
corepack pnpm run benchmark:asr:mlx
```

TTS 自动结构校验不能替代人工听感确认。以下三段已在 ElectronQwenSpeech 中
实际生成并验证播放控件，主观听感仍待人工确认：

- 多音字：`银行行长走过人行道。`；
- 数字日期：`今天是2026年8月13日，下午3点30分。`；
- 带指令对白：`“别着急，”她轻声说，“我们马上出发。”`。

本分支仅支持开发运行，不提供打包、签名、发布或模型分发。

## 验证报告与迁移来源

历史验证数据、方法和当前待确认项见
[MLX BF16 完整语音栈验证报告](./docs/MLX语音栈验证报告-2026-08-13.md)。报告中的
MPS 对照数据仅记录源仓历史基准，不表示本分支包含或可执行 MPS runtime。
