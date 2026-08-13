# ElectronQwenSpeech

ElectronQwenSpeech 是独立的 Qwen 本地语音诊断项目。仓库使用运行时分支隔离不同实现，`main` 只维护分支说明，不承载可运行代码。

## 分支

| 分支 | 状态 | 说明 |
| --- | --- | --- |
| `main` | 说明分支 | 维护分支用途与迁移来源。 |
| `mps` | 当前实现 | PyTorch/MPS 语音诊断实现，使用 `mps:0`、`bfloat16`，禁止 CPU fallback。 |
| `mlx` | 尚未创建 | 待源仓 MLX 调试完成后再独立迁移。 |

切换到当前可运行实现：

```bash
git switch mps
```

## MPS 实现边界

`mps` 是当前唯一可运行实现，固定使用 PyTorch、`mps:0` 与 `bfloat16`，并将
`PYTORCH_ENABLE_MPS_FALLBACK` 固定为 `0`。项目不提供 CPU fallback、运行时切换
或 MLX 依赖；模型文件也不纳入仓库。

模型目录通过根目录 `.env.local` 配置，不在实现代码中固定机器路径。先复制模板：

```bash
cp .env.example .env.local
```

然后把三项值改为本机模型的绝对目录：

```dotenv
ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH=/absolute/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH=/absolute/path/to/Qwen3-ASR-1.7B
ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH=/absolute/path/to/Qwen3-ForcedAligner-0.6B
```

`.env.local` 已忽略追踪；`.env.example` 只保存可提交的占位符。根目录的 `dev` 和
三类真实测试命令会自动加载这三项变量，不会转发 `.env.local` 中的其它变量；如果
Shell 已设置同名变量，则 Shell 的值优先。

## 环境与安装

- Node.js `24.13.0`（见 `.nvmrc`）
- pnpm `11.5.2`（由 `package.json` 的 `packageManager` 固定）
- Python `3.12`

安装 Node.js workspace 依赖：

```bash
fnm use
corepack pnpm install
```

ASR 与 TTS 使用两个相互隔离的 Python 环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r services/asr-sidecar/requirements.txt

python3.12 -m venv services/tts-sidecar/.venv
services/tts-sidecar/.venv/bin/python -m pip install -r services/tts-sidecar/requirements.txt
```

## 开发与验证

启动 Electron 语音诊断台：

```bash
corepack pnpm run dev
```

执行静态检查和普通自动化测试：

```bash
corepack pnpm run check
```

分别执行真实 ASR、真实 TTS 与完整语音闭环测试：

```bash
corepack pnpm run test:asr:real
corepack pnpm run test:tts:real
corepack pnpm run test:speech:real
```

真实测试直接加载 `.env.local` 配置的本机模型，并要求 MPS 可用。该仓库仅支持开发运行，根脚本
不提供 `package` 或 `make`，Electron Forge 配置也会拒绝生成安装包。
