// @vitest-environment node

import type { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';

import { Buffer } from 'node:buffer';
import { EventEmitter } from 'node:events';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import { TTSProviderError } from '@electron-qwen-speech/application';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { QWEN_LOCAL_TTS_DEFAULTS, QwenLocalTTSProvider } from './qwenLocalTTSProvider.ts';

interface CapturedRequest {
  id: string;
  method: string;
  params: Record<string, unknown>;
}

const TEST_TTS_MODEL_PATH = '/models/mlx/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16';

class FakeSidecarProcess extends EventEmitter {
  readonly stdin = new PassThrough();
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  readonly requests: CapturedRequest[] = [];
  readonly killSignals: Array<NodeJS.Signals | number | undefined> = [];
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  onRequest: ((request: CapturedRequest) => void) | undefined;
  #stdinBuffer = '';

  constructor() {
    super();
    this.stdin.setEncoding('utf8');
    this.stdin.on('data', (chunk: string) => {
      this.#stdinBuffer += chunk;
      let newlineIndex = this.#stdinBuffer.indexOf('\n');
      while (newlineIndex >= 0) {
        const line = this.#stdinBuffer.slice(0, newlineIndex);
        this.#stdinBuffer = this.#stdinBuffer.slice(newlineIndex + 1);
        const request = JSON.parse(line) as CapturedRequest;
        this.requests.push(request);
        this.onRequest?.(request);
        this.emit('request');
        newlineIndex = this.#stdinBuffer.indexOf('\n');
      }
    });
  }

  kill(signal?: NodeJS.Signals | number): boolean {
    this.killSignals.push(signal);
    return true;
  }

  finish(code: number | null, signal: NodeJS.Signals | null = null): void {
    this.exitCode = code;
    this.signalCode = signal;
    this.emit('exit', code, signal);
    this.emit('close', code, signal);
  }

  respond(id: string, result: unknown): void {
    this.stdout.write(`${JSON.stringify({ id, result })}\n`);
  }

  reject(id: string, code: string, message: string): void {
    this.stdout.write(`${JSON.stringify({ id, error: { code, message, retryable: false } })}\n`);
  }

  async waitForRequests(count: number): Promise<void> {
    while (this.requests.length < count)
      await EventEmitter.once(this, 'request');
  }
}

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map(directory => rm(directory, {
    force: true,
    recursive: true,
  })));
});

function createWave(sampleRate = 24000, channels = 1, frameCount = 2400): Buffer {
  const dataSize = frameCount * channels * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write('RIFF', 0, 'ascii');
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8, 'ascii');
  buffer.write('fmt ', 12, 'ascii');
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(channels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * channels * 2, 28);
  buffer.writeUInt16LE(channels * 2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36, 'ascii');
  buffer.writeUInt32LE(dataSize, 40);
  return buffer;
}

function statusResult(state = 'uninitialized') {
  return {
    state,
    initialized: state === 'ready',
    currentTaskId: null,
    queuedTaskCount: 0,
    cancelRequested: false,
    loadCount: state === 'ready' ? 1 : 0,
  };
}

function initializeResult() {
  const runtime = QWEN_LOCAL_TTS_DEFAULTS;
  return {
    state: 'ready',
    runtime: runtime.runtime,
    accelerator: runtime.accelerator,
    modelPath: TEST_TTS_MODEL_PATH,
    device: runtime.device,
    modelFormat: runtime.modelFormat,
    modelRevision: runtime.modelRevision,
    modelPrecision: runtime.modelPrecision,
    speechTokenizerPrecision: runtime.speechTokenizerPrecision,
    dtype: 'bfloat16',
    supportedLanguages: ['chinese', 'english'],
    supportedVoiceIds: ['vivian', 'ryan'],
    loadCount: 1,
  };
}

function createProvider(
  process: FakeSidecarProcess,
  timeoutMs = 100,
): { provider: QwenLocalTTSProvider; outputDirectory: string } {
  const outputDirectory = mkdtempSync(path.join(os.tmpdir(), 'electron-qwen-speech-tts-provider-test-'));
  temporaryDirectories.push(outputDirectory);
  const provider = new QwenLocalTTSProvider({
    environment: {
      ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH: TEST_TTS_MODEL_PATH,
    },
    pythonExecutable: '/fake/python',
    sidecarEntryPoint: '/fake/main.py',
    workingDirectory: '/fake',
    outputDirectory,
    spawnProcess: (() => process as unknown as ChildProcessWithoutNullStreams) as unknown as typeof spawn,
    timeouts: {
      controlMs: timeoutMs,
      initializeMs: timeoutMs,
      synthesizeMs: timeoutMs,
      shutdownMs: timeoutMs,
    },
  });
  return { provider, outputDirectory };
}

describe('qwenLocalTTSProvider', () => {
  it('解析初始化 capability 与 UTF-8 分块响应', async () => {
    const process = new FakeSidecarProcess();
    const { provider } = createProvider(process);
    const initialization = provider.initialize();
    await process.waitForRequests(1);
    const request = process.requests[0]!;
    const line = Buffer.from(`${JSON.stringify({ id: request.id, result: initializeResult() })}\n`);
    process.stdout.write(line.subarray(0, 9));
    process.stdout.write(line.subarray(9));

    await expect(initialization).resolves.toMatchObject({
      runtime: 'mlx',
      accelerator: 'metal',
      device: 'gpu:0',
      modelPrecision: 'bfloat16',
      speechTokenizerPrecision: 'float32',
      supportedVoiceIds: ['vivian', 'ryan'],
    });
    process.finish(0);
  });

  it('固定使用 MLX Python、本地模型路径与离线环境', async () => {
    const process = new FakeSidecarProcess();
    process.onRequest = (request) => {
      if (request.method === 'health')
        process.respond(request.id, { ...statusResult(), healthy: true });
    };
    const outputDirectory = mkdtempSync(path.join(os.tmpdir(), 'electron-qwen-speech-tts-provider-test-'));
    temporaryDirectories.push(outputDirectory);
    const spawnProcess = vi.fn(() => process as unknown as ChildProcessWithoutNullStreams) as unknown as typeof spawn;
    const provider = new QwenLocalTTSProvider({
      environment: {
        ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH: TEST_TTS_MODEL_PATH,
      },
      outputDirectory,
      spawnProcess,
    });

    await provider.health();

    expect(spawnProcess).toHaveBeenCalledWith(
      QWEN_LOCAL_TTS_DEFAULTS.pythonExecutable,
      ['-u', QWEN_LOCAL_TTS_DEFAULTS.sidecarEntryPoint],
      expect.objectContaining({
        env: expect.objectContaining({
          HF_DATASETS_OFFLINE: '1',
          HF_HUB_DISABLE_TELEMETRY: '1',
          HF_HUB_OFFLINE: '1',
          TRANSFORMERS_OFFLINE: '1',
          ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH: TEST_TTS_MODEL_PATH,
        }),
      }),
    );
    process.finish(0);
  });

  it('缺少 TTS 模型路径环境变量时拒绝创建 Provider', () => {
    expect(() => new QwenLocalTTSProvider({
      environment: {
        ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH: '',
      },
    })).toThrowError(expect.objectContaining({
      code: 'MODEL_INTEGRITY_FAILED',
    }));
  });

  it('将自定义 TTS 模型路径传递给 Sidecar', async () => {
    const process = new FakeSidecarProcess();
    process.onRequest = (request) => {
      if (request.method === 'health')
        process.respond(request.id, { ...statusResult(), healthy: true });
    };
    const outputDirectory = mkdtempSync(path.join(os.tmpdir(), 'electron-qwen-speech-tts-provider-test-'));
    const customModelPath = '/opt/electron-qwen-speech/models/tts';
    temporaryDirectories.push(outputDirectory);
    const spawnProcess = vi.fn(() => process as unknown as ChildProcessWithoutNullStreams) as unknown as typeof spawn;
    const provider = new QwenLocalTTSProvider({
      environment: {
        ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH: customModelPath,
      },
      outputDirectory,
      spawnProcess,
    });

    await provider.health();

    expect(spawnProcess).toHaveBeenCalledWith(
      expect.any(String),
      expect.any(Array),
      expect.objectContaining({
        env: expect.objectContaining({
          ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH: customModelPath,
        }),
      }),
    );
    process.finish(0);
  });

  it('拒绝加速器和固定模型身份不一致的初始化元数据', async () => {
    const runtimeProcess = new FakeSidecarProcess();
    const { provider: runtimeProvider } = createProvider(runtimeProcess);
    const runtimeInitialization = runtimeProvider.initialize();
    await runtimeProcess.waitForRequests(1);
    runtimeProcess.respond(runtimeProcess.requests[0]!.id, {
      ...initializeResult(),
      accelerator: 'cpu',
    });
    await expect(runtimeInitialization).rejects.toMatchObject({ code: 'RUNTIME_DEVICE_MISMATCH' });
    runtimeProcess.finish(0);

    const revisionProcess = new FakeSidecarProcess();
    const { provider: revisionProvider } = createProvider(revisionProcess);
    const revisionInitialization = revisionProvider.initialize();
    await revisionProcess.waitForRequests(1);
    revisionProcess.respond(revisionProcess.requests[0]!.id, {
      ...initializeResult(),
      modelRevision: 'unexpected',
    });
    await expect(revisionInitialization).rejects.toMatchObject({ code: 'MODEL_INTEGRITY_FAILED' });
    revisionProcess.finish(0);
  });

  it('合成请求使用会话输出目录并校验 PCM WAV', async () => {
    const process = new FakeSidecarProcess();
    const { provider, outputDirectory } = createProvider(process);
    const synthesis = provider.synthesize({
      taskId: 'polyphone',
      text: '银行行长走过人行道。',
      language: 'Chinese',
      voiceId: 'Vivian',
      instruction: '自然朗读',
    });
    await process.waitForRequests(1);
    const request = process.requests[0]!;
    const audioPath = request.params.outputPath as string;
    expect(path.dirname(audioPath)).toBe(outputDirectory);
    expect(request).toMatchObject({
      method: 'synthesize',
      params: {
        taskId: 'polyphone',
        language: 'Chinese',
        voiceId: 'Vivian',
      },
    });
    writeFileSync(audioPath, createWave());
    process.respond(request.id, {
      taskId: 'polyphone',
      audioPath,
      mimeType: 'audio/wav',
      sampleRate: 24000,
      channels: 1,
      frameCount: 2400,
      durationSeconds: 0.1,
      generationSeconds: 0.25,
    });

    await expect(synthesis).resolves.toMatchObject({
      taskId: 'polyphone',
      sampleRate: 24000,
      durationSeconds: 0.1,
    });
    process.finish(0);
  });

  it('拒绝远端错误、响应路径不一致和损坏 WAV', async () => {
    const remoteProcess = new FakeSidecarProcess();
    const { provider: remoteProvider } = createProvider(remoteProcess);
    const remoteSynthesis = remoteProvider.synthesize({
      taskId: 'unsupported',
      text: '测试',
      language: 'Chinese',
      voiceId: 'Unknown',
    });
    await remoteProcess.waitForRequests(1);
    remoteProcess.reject(remoteProcess.requests[0]!.id, 'UNSUPPORTED_VOICE', '音色不支持。');
    const remoteError = await remoteSynthesis.catch((error: unknown) => error);
    expect(remoteError).toBeInstanceOf(TTSProviderError);
    expect(remoteError).toMatchObject({ code: 'UNSUPPORTED_VOICE' });
    remoteProcess.finish(0);

    const corruptProcess = new FakeSidecarProcess();
    const { provider: corruptProvider } = createProvider(corruptProcess);
    const corruptSynthesis = corruptProvider.synthesize({
      taskId: 'corrupt',
      text: '测试',
      language: 'Chinese',
      voiceId: 'Vivian',
    });
    await corruptProcess.waitForRequests(1);
    const request = corruptProcess.requests[0]!;
    const audioPath = request.params.outputPath as string;
    writeFileSync(audioPath, Buffer.from('not-wave'));
    corruptProcess.respond(request.id, {
      taskId: 'corrupt',
      audioPath,
      mimeType: 'audio/wav',
      sampleRate: 24000,
      channels: 1,
      frameCount: 1,
      durationSeconds: 1 / 24000,
      generationSeconds: 0.1,
    });
    await expect(corruptSynthesis).rejects.toMatchObject({ code: 'AUDIO_OUTPUT_INVALID' });
    corruptProcess.finish(0);
  });

  it('映射超时、异常退出与取消响应', async () => {
    const timeoutProcess = new FakeSidecarProcess();
    const { provider: timeoutProvider } = createProvider(timeoutProcess, 20);
    await expect(timeoutProvider.health()).rejects.toMatchObject({ code: 'REQUEST_TIMEOUT' });
    timeoutProcess.finish(0);

    const exitProcess = new FakeSidecarProcess();
    const { provider: exitProvider } = createProvider(exitProcess);
    const health = exitProvider.health();
    await exitProcess.waitForRequests(1);
    exitProcess.stderr.write('tts traceback');
    exitProcess.finish(9);
    await expect(health).rejects.toMatchObject({
      code: 'SIDECAR_EXITED',
      details: { code: 9, stderr: 'tts traceback' },
    });

    const cancelProcess = new FakeSidecarProcess();
    const { provider: cancelProvider } = createProvider(cancelProcess);
    const cancellation = cancelProvider.cancel('tts-task');
    await cancelProcess.waitForRequests(1);
    cancelProcess.respond(cancelProcess.requests[0]!.id, {
      taskId: 'tts-task',
      status: 'cancel_requested',
    });
    await expect(cancellation).resolves.toEqual({
      taskId: 'tts-task',
      status: 'cancel_requested',
    });
    cancelProcess.finish(0);
  });

  it('dispose 发送 shutdown 并等待退出', async () => {
    const process = new FakeSidecarProcess();
    process.onRequest = (request) => {
      if (request.method === 'health')
        process.respond(request.id, { ...statusResult(), healthy: true });
      if (request.method === 'shutdown') {
        process.respond(request.id, { state: 'shutting_down' });
        queueMicrotask(() => process.finish(0));
      }
    };
    const { provider } = createProvider(process);
    await provider.health();

    await expect(provider.dispose()).resolves.toBeUndefined();
    await expect(provider.getStatus()).resolves.toMatchObject({ state: 'disposed' });
    expect(process.killSignals).toEqual([]);
  });
});
