// @vitest-environment node

import type { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';

import { Buffer } from 'node:buffer';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { ASRProviderError } from '@electron-qwen-speech/application';
import { describe, expect, it } from 'vitest';
import { QwenLocalASRProvider } from './qwenLocalASRProvider.ts';

const TEST_ALIGNER_MODEL_PATH = '/models/Qwen3-ForcedAligner-0.6B';
const TEST_ASR_MODEL_PATH = '/models/Qwen3-ASR-1.7B';

interface CapturedRequest {
  id: string;
  method: string;
  params: Record<string, unknown>;
}

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

function createProvider(
  process: FakeSidecarProcess,
  timeoutMs = 100,
): QwenLocalASRProvider {
  return new QwenLocalASRProvider({
    environment: {
      ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH: TEST_ALIGNER_MODEL_PATH,
      ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH: TEST_ASR_MODEL_PATH,
    },
    pythonExecutable: '/fake/python',
    sidecarEntryPoint: '/fake/main.py',
    workingDirectory: '/fake',
    spawnProcess: (() => process as unknown as ChildProcessWithoutNullStreams) as unknown as typeof spawn,
    timeouts: {
      controlMs: timeoutMs,
      initializeMs: timeoutMs,
      transcribeMs: timeoutMs,
      shutdownMs: timeoutMs,
    },
  });
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

describe('qwenLocalASRProvider', () => {
  it('要求通过环境变量配置两个绝对模型路径', () => {
    expect(() => new QwenLocalASRProvider({
      environment: {
        ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH: '',
        ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH: '',
      },
    })).toThrowError(ASRProviderError);
  });

  it('按 JSON Lines 拆包并按请求 id 关联乱序响应', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process);

    const healthPromise = provider.health();
    const statusPromise = provider.getStatus();
    await process.waitForRequests(2);

    const healthRequest = process.requests.find(request => request.method === 'health')!;
    const statusRequest = process.requests.find(request => request.method === 'status')!;
    process.respond(statusRequest.id, statusResult());

    const healthLine = Buffer.from(`${JSON.stringify({
      id: healthRequest.id,
      result: { ...statusResult(), healthy: true },
    })}\n`);
    process.stdout.write(healthLine.subarray(0, 7));
    process.stdout.write(healthLine.subarray(7, 19));
    process.stdout.write(healthLine.subarray(19));

    await expect(statusPromise).resolves.toMatchObject({ state: 'uninitialized' });
    await expect(healthPromise).resolves.toMatchObject({ healthy: true });
    process.finish(0);
  });

  it('正确拆分跨 chunk 的 UTF-8 初始化响应', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process);
    const initializePromise = provider.initialize();
    await process.waitForRequests(1);

    const request = process.requests[0]!;
    const line = Buffer.from(`${JSON.stringify({
      id: request.id,
      result: {
        state: 'ready',
        asrModelPath: TEST_ASR_MODEL_PATH,
        alignerModelPath: TEST_ALIGNER_MODEL_PATH,
        asrDevice: 'mps:0',
        alignerDevice: 'mps:0',
        asrDtype: 'bfloat16',
        alignerDtype: 'bfloat16',
        loadCount: 1,
      },
    })}\n`);
    const multibyteOffset = line.indexOf(Buffer.from('模')) + 1;
    process.stdout.write(line.subarray(0, multibyteOffset));
    process.stdout.write(line.subarray(multibyteOffset));

    await expect(initializePromise).resolves.toMatchObject({
      state: 'ready',
      asrDevice: 'mps:0',
      alignerDevice: 'mps:0',
    });
    process.finish(0);
  });

  it('把控制请求超时映射为 ASRProviderError', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process, 20);

    await expect(provider.health()).rejects.toMatchObject({
      name: 'ASRProviderError',
      code: 'REQUEST_TIMEOUT',
    });
    process.finish(0);
  });

  it('把非法 JSON 视为协议错误并终止失去同步的 Sidecar', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process);
    const healthPromise = provider.health();
    await process.waitForRequests(1);

    process.stdout.write('{not-json}\n');

    await expect(healthPromise).rejects.toMatchObject({ code: 'SIDECAR_PROTOCOL_ERROR' });
    expect(process.killSignals).toEqual(['SIGTERM']);
    process.finish(null, 'SIGTERM');
  });

  it('把结构化 Python 错误映射为 ASRProviderError', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process);
    const transcription = provider.transcribe({
      taskId: 'missing-audio',
      audioPath: '/missing.wav',
    });
    await process.waitForRequests(1);
    process.reject(process.requests[0]!.id, 'AUDIO_NOT_FOUND', '音频不存在。');

    const error = await transcription.catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ASRProviderError);
    expect(error).toMatchObject({ code: 'AUDIO_NOT_FOUND', retryable: false });
    process.finish(0);
  });

  it('保持单条 transcribe 接口兼容', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process);
    const transcription = provider.transcribe({
      taskId: 'single-task',
      audioPath: '/single.wav',
    });
    await process.waitForRequests(1);
    expect(process.requests[0]).toMatchObject({
      method: 'transcribe',
      params: { taskId: 'single-task', audioPath: '/single.wav' },
    });
    process.respond(process.requests[0]!.id, {
      taskId: 'single-task',
      text: '你好',
      language: 'Chinese',
      timestamps: [{ text: '你好', start: 0.1, end: 0.4 }],
    });

    await expect(transcription).resolves.toMatchObject({
      taskId: 'single-task',
      text: '你好',
      language: 'Chinese',
    });
    process.finish(0);
  });

  it('发送一个批量协议请求并按输入顺序验证响应', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process);
    const transcription = provider.transcribeBatch({
      taskId: 'batch-task',
      items: [
        { itemId: 'first', audioPath: '/first.wav' },
        { itemId: 'second', audioPath: '/second.mp3', language: 'Chinese' },
      ],
    });
    await process.waitForRequests(1);
    expect(process.requests[0]).toMatchObject({
      method: 'transcribe',
      params: {
        taskId: 'batch-task',
        items: [
          { itemId: 'first', audioPath: '/first.wav' },
          { itemId: 'second', audioPath: '/second.mp3', language: 'Chinese' },
        ],
      },
    });
    process.respond(process.requests[0]!.id, {
      taskId: 'batch-task',
      items: [
        {
          itemId: 'first',
          text: '一',
          language: 'Chinese',
          timestamps: [{ text: '一', start: 0, end: 0.2 }],
        },
        {
          itemId: 'second',
          text: '二',
          language: 'Chinese',
          timestamps: [{ text: '二', start: 0.2, end: 0.4 }],
        },
      ],
    });

    await expect(transcription).resolves.toMatchObject({
      taskId: 'batch-task',
      items: [
        { itemId: 'first', text: '一' },
        { itemId: 'second', text: '二' },
      ],
    });
    process.finish(0);
  });

  it('拒绝重复批量 itemId 与错序响应', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process);
    await expect(provider.transcribeBatch({
      taskId: 'duplicate-items',
      items: [
        { itemId: 'same', audioPath: '/first.wav' },
        { itemId: 'same', audioPath: '/second.wav' },
      ],
    })).rejects.toMatchObject({ code: 'INVALID_REQUEST' });

    const transcription = provider.transcribeBatch({
      taskId: 'wrong-order',
      items: [
        { itemId: 'first', audioPath: '/first.wav' },
        { itemId: 'second', audioPath: '/second.wav' },
      ],
    });
    await process.waitForRequests(1);
    process.respond(process.requests[0]!.id, {
      taskId: 'wrong-order',
      items: [
        { itemId: 'second', text: '二', timestamps: [] },
        { itemId: 'first', text: '一', timestamps: [] },
      ],
    });

    await expect(transcription).rejects.toMatchObject({ code: 'SIDECAR_PROTOCOL_ERROR' });
    process.finish(0);
  });

  it('批量转写使用转写超时并支持整批取消', async () => {
    const timeoutProcess = new FakeSidecarProcess();
    const timeoutProvider = createProvider(timeoutProcess, 20);
    await expect(timeoutProvider.transcribeBatch({
      taskId: 'slow-batch',
      items: [{ itemId: 'only', audioPath: '/slow.wav' }],
    })).rejects.toMatchObject({ code: 'REQUEST_TIMEOUT' });
    timeoutProcess.finish(0);

    const cancelProcess = new FakeSidecarProcess();
    const cancelProvider = createProvider(cancelProcess);
    const cancellation = cancelProvider.cancel('batch-task');
    await cancelProcess.waitForRequests(1);
    expect(cancelProcess.requests[0]).toMatchObject({
      method: 'cancel',
      params: { taskId: 'batch-task' },
    });
    cancelProcess.respond(cancelProcess.requests[0]!.id, {
      taskId: 'batch-task',
      status: 'cancel_requested',
    });
    await expect(cancellation).resolves.toEqual({
      taskId: 'batch-task',
      status: 'cancel_requested',
    });
    cancelProcess.finish(0);
  });

  it('sidecar 退出时拒绝所有未完成请求', async () => {
    const process = new FakeSidecarProcess();
    const provider = createProvider(process);
    const transcription = provider.transcribe({
      taskId: 'running-task',
      audioPath: '/audio.wav',
    });
    await process.waitForRequests(1);

    process.stderr.write('traceback on stderr');
    process.finish(17);

    await expect(transcription).rejects.toMatchObject({
      code: 'SIDECAR_EXITED',
      details: {
        code: 17,
        stderr: 'traceback on stderr',
      },
    });
  });

  it('dispose 发送 shutdown 并等待正常退出', async () => {
    const process = new FakeSidecarProcess();
    process.onRequest = (request) => {
      if (request.method === 'health')
        process.respond(request.id, { ...statusResult(), healthy: true });
      if (request.method === 'shutdown') {
        process.respond(request.id, { state: 'shutting_down' });
        queueMicrotask(() => process.finish(0));
      }
    };
    const provider = createProvider(process);
    await provider.health();

    await expect(provider.dispose()).resolves.toBeUndefined();
    await expect(provider.getStatus()).resolves.toMatchObject({ state: 'disposed' });
    expect(process.killSignals).toEqual([]);
  });

  it('退出超时后只终止自己创建的子进程', async () => {
    const process = new FakeSidecarProcess();
    process.onRequest = (request) => {
      if (request.method === 'health')
        process.respond(request.id, { ...statusResult(), healthy: true });
    };
    const provider = createProvider(process, 20);
    await provider.health();

    await expect(provider.dispose()).rejects.toMatchObject({ code: 'SHUTDOWN_TIMEOUT' });
    expect(process.killSignals).toEqual(['SIGTERM']);
  });
});
