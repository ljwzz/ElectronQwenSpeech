# Repository Guidelines

## Project Scope

ElectronQwenSpeech is a development-only Electron diagnostic application for local
Qwen ASR and TTS on Apple MPS. The `mps` branch is the only runnable implementation.
Do not add a second runtime implementation, runtime selectors, parallel dependency sets,
or compatibility branches. Another runtime will be migrated separately after its source
implementation is stable.

The MPS baseline and immutable source commit are recorded in `README.md`. Do not copy
later source-worktree changes into this repository without an explicit migration decision.

## Project Structure

`apps/desktop/` contains the Electron application: `main/` owns processes and IPC,
`preload/` exposes the restricted bridge, and `renderer/src/` contains the Vue UI.
`packages/application/` defines ASR/TTS provider contracts, while `packages/contracts/`
contains shared presentation contracts. Python JSONL sidecars live in
`services/asr-sidecar/` and `services/tts-sidecar/`; keep their tests and fixtures with
their owning service.

Do not add product workbench, persistence, project-management, publishing, signing, or
installer responsibilities. Generated Forge output under `apps/desktop/.vite/` and
`apps/desktop/out/` must not be edited.

## Runtime Invariants

Use Python 3.12, `mps:0`, `bfloat16`, and `PYTORCH_ENABLE_MPS_FALLBACK=0`. Read the ASR,
Forced Aligner, and TTS model paths from the three `ELECTRON_QWEN_SPEECH_*_MODEL_PATH`
environment variables and fail when they are missing, relative, or unavailable; never
silently fall back to CPU. Keep local values in ignored `.env.local`, keep only placeholders
in `.env.example`, and never commit a developer-specific absolute model path.

Sidecar stdout is reserved for JSON Lines protocol responses. Send diagnostics and
tracebacks to stderr. Preserve the existing initialize, health, status, operation,
cancel, and shutdown semantics. Keep the ASR environment at repository root `.venv/`
and the TTS environment at `services/tts-sidecar/.venv/`.

Maintain Electron isolation boundaries: expose individual preload methods, validate IPC
senders and main frames, deny unexpected navigation and permissions, and keep renderer
Node.js integration disabled.

## Commands

Use Node.js `v24.13.0` and pnpm `11.5.2`.

- `corepack pnpm install` installs workspace dependencies.
- `corepack pnpm run dev` starts the diagnostic application.
- `corepack pnpm run check` runs lint, type checks, and ordinary tests.
- `corepack pnpm run test:asr:real` runs the real ASR and aligner check.
- `corepack pnpm run test:tts:real` runs the real TTS check.
- `corepack pnpm run test:speech:real` runs the real TTS-to-ASR loop.

Do not add or use packaging and distribution commands. Model files are local prerequisites
and must not be copied into the repository.

## Code and Test Conventions

Use two-space indentation, LF endings, a final newline, single quotes, semicolons, and
1TBS braces. Vue SFC blocks are ordered `script`, `template`, then `style`; component
names use PascalCase. Keep TypeScript modules camelCase and tests beside their subjects
as `*.test.ts` or `*.real.ts`.

Add focused tests for changed behavior. Run the narrow test while iterating, then run the
root `check`; run real-model tests only when the local models and both Python environments
are available. Do not describe static checks as proof of real MPS or UI behavior.

Use lowercase Conventional Commit types and keep each commit scoped to one concern. Do
not commit virtual environments, model data, generated output, secrets, or unrelated
source-repository changes.
