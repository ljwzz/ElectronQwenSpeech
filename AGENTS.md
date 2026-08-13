# Repository Guidelines

## Project Scope

ElectronQwenSpeech is a development-only Electron diagnostic application for local
Qwen ASR and TTS on Apple MLX/Metal. The `mlx` branch contains one MLX runtime and
must not add runtime selectors, alternate backend dependencies, compatibility branches,
or automatic fallback.

The immutable implementation and report sources are commits
`b86f9ce1adf6a6a8863efd2067e4c791c386d4f6` and
`04c7a61313f9b4e273e90848009aff579b08b994` in the source repository recorded in
`README.md`. Do not copy later source-worktree changes without an explicit migration
decision.

## Project Structure

`apps/desktop/` contains the Electron application: `main/` owns processes and IPC,
`preload/` exposes the restricted bridge, and `renderer/src/` contains the Vue UI.
`packages/application/` defines ASR/TTS provider contracts, while `packages/contracts/`
contains shared presentation contracts. Python JSONL sidecars live in
`services/asr-sidecar/` and `services/tts-sidecar/`; keep their tests and fixtures with
their owning service. Model integrity and benchmark tooling lives in `scripts/`.

Do not add product workbench, persistence, project-management, publishing, signing, or
installer responsibilities. Generated Forge output under `apps/desktop/.vite/` and
`apps/desktop/out/` must not be edited.

## Runtime Invariants

Use Python 3.12, MLX/Metal `gpu:0`, BF16 main model weights, and the source model
revisions fixed in `scripts/mlx_speech_model_manifest.json`. The TTS speech tokenizer
remains FP32. Fail when MLX, Metal, a model revision, or its integrity check is
unavailable; never switch to another model backend.

Configure the three absolute local model directories in the ignored root `.env.local`;
keep only placeholders in `.env.example`. Never add machine-specific model paths to
tracked source, manifests, tests, or documentation. Root runtime and real-model commands
load `.env.local`; inherited process variables take precedence.

Sidecar stdout is reserved for JSON Lines protocol responses. Send diagnostics and
tracebacks to stderr. Preserve initialize, health, status, operation, cancel, and
shutdown semantics. Use `services/asr-sidecar/.venv-mlx/` for ASR/Forced Aligner and
`services/tts-sidecar/.venv-mlx/` for TTS. Each sidecar owns one `requirements.txt`.

Maintain Electron isolation boundaries: expose individual preload methods, validate IPC
senders and main frames, deny unexpected navigation and permissions, and keep renderer
Node.js integration disabled.

## Commands

Use Node.js `v24.13.0`, pnpm `11.5.2`, and Python 3.12.

- `corepack pnpm install` installs workspace dependencies.
- `corepack pnpm run dev` starts the MLX diagnostic application.
- `corepack pnpm run check` runs lint, type checks, ordinary tests, and script tests.
- `corepack pnpm run validate:speech-models:mlx` validates all fixed model files.
- `corepack pnpm run test:asr:real` runs the real MLX ASR and aligner check.
- `corepack pnpm run test:tts:real` runs the real MLX TTS check.
- `corepack pnpm run test:speech:real` runs the real MLX TTS-to-ASR loop.
- `corepack pnpm run benchmark:tts:mlx` runs the TTS benchmark.
- `corepack pnpm run benchmark:asr:mlx` runs the ASR/Aligner benchmark.

Do not add or use packaging and distribution commands. Model files are local
prerequisites and must not be copied into the repository.

## Code and Test Conventions

Use two-space indentation, LF endings, a final newline, single quotes, semicolons, and
1TBS braces. Vue SFC blocks are ordered `script`, `template`, then `style`; component
names use PascalCase. Keep TypeScript modules camelCase and tests beside their subjects
as `*.test.ts` or `*.real.ts`.

Add focused tests for changed behavior. Run the narrow test while iterating, then run the
root `check`; run real-model tests and benchmarks only when the local models and both
Python environments are available. Do not describe static checks as proof of real MLX,
UI behavior, or the three pending TTS listening checks.

Use lowercase Conventional Commit types and keep each commit scoped to one concern. Do
not commit virtual environments, model data, generated output, secrets, or unrelated
source-repository changes.
