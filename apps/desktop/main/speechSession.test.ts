// @vitest-environment node

import type {
  ASRBatchProvider,
  ASRBatchTranscriptionRequest,
  ASRBatchTranscriptionResult,
  ASRCancelResult,
  ASRInitializeResult,
  ASRProviderHealth,
  ASRProviderStatus,
  ASRTranscriptionRequest,
  ASRTranscriptionResult,
} from '@electron-qwen-speech/application';
import type { RunnableSpeechFixture } from './fixtureCatalog.ts';

import { ASRProviderError } from '@electron-qwen-speech/application';
import { describe, expect, it, vi } from 'vitest';
import { SPEECH_LEASE_DURATIONS_MS, SpeechSession } from './speechSession.ts';

interface FakeTimer {
  id: number;
  deadline: number;
  handler: () => void;
}

class FakeClock {
  currentTime = 0;
  #sequence = 0;
  #timers = new Map<number, FakeTimer>();

  now = (): number => this.currentTime;

  setTimeout = (handler: () => void, delayMs: number): number => {
    const id = ++this.#sequence;
    this.#timers.set(id, { id, deadline: this.currentTime + delayMs, handler });
    return id;
  };

  clearTimeout = (handle: unknown): void => {
    this.#timers.delete(handle as number);
  };

  advance(milliseconds: number): void {
    this.currentTime += milliseconds;
    while (true) {
      const due = [...this.#timers.values()]
        .filter(timer => timer.deadline <= this.currentTime)
        .sort((left, right) => left.deadline - right.deadline || left.id - right.id)[0];
      if (!due)
        return;
      this.#timers.delete(due.id);
      due.handler();
    }
  }
}

function initializationResult(): ASRInitializeResult {
  return {
    state: 'ready',
    runtime: 'mlx',
    accelerator: 'metal',
    device: 'gpu:0',
    modelFormat: 'mlx',
    asrModelPath: '/models/asr',
    alignerModelPath: '/models/aligner',
    asrModelRevision: 'asr-revision',
    alignerModelRevision: 'aligner-revision',
    asrModelPrecision: 'bfloat16',
    alignerModelPrecision: 'bfloat16',
    asrDevice: 'gpu:0',
    alignerDevice: 'gpu:0',
    asrDtype: 'bfloat16',
    alignerDtype: 'bfloat16',
    loadCount: 1,
  };
}

function readyStatus(): ASRProviderStatus {
  return {
    state: 'ready',
    initialized: true,
    currentTaskId: null,
    queuedTaskCount: 0,
    cancelRequested: false,
    loadCount: 1,
  };
}

class FakeProvider implements ASRBatchProvider {
  initializeCalls = 0;
  disposeCalls = 0;
  getStatusCalls = 0;
  status: ASRProviderStatus = readyStatus();
  transcribeCalls: ASRTranscriptionRequest[] = [];
  batchCalls: ASRBatchTranscriptionRequest[] = [];
  cancelCalls: string[] = [];
  onInitialize: (() => Promise<ASRInitializeResult>) | undefined;
  onTranscribe: ((request: ASRTranscriptionRequest) => Promise<ASRTranscriptionResult>) | undefined;
  onBatch: ((request: ASRBatchTranscriptionRequest) => Promise<ASRBatchTranscriptionResult>) | undefined;
  onCancel: ((taskId: string) => Promise<ASRCancelResult>) | undefined;

  async initialize(): Promise<ASRInitializeResult> {
    this.initializeCalls += 1;
    return this.onInitialize ? this.onInitialize() : initializationResult();
  }

  async health(): Promise<ASRProviderHealth> {
    return { ...readyStatus(), healthy: true };
  }

  async transcribe(request: ASRTranscriptionRequest): Promise<ASRTranscriptionResult> {
    this.transcribeCalls.push(request);
    if (this.onTranscribe)
      return this.onTranscribe(request);
    return {
      taskId: request.taskId,
      text: '你好',
      language: 'Chinese',
      timestamps: [{ text: '你好', start: 0, end: 0.5 }],
      alignmentStatus: 'aligned',
    };
  }

  async transcribeBatch(request: ASRBatchTranscriptionRequest): Promise<ASRBatchTranscriptionResult> {
    this.batchCalls.push(request);
    if (this.onBatch)
      return this.onBatch(request);
    return {
      taskId: request.taskId,
      items: request.items.map(item => ({
        itemId: item.itemId,
        text: '你好',
        language: 'Chinese',
        timestamps: [{ text: '你好', start: 0, end: 0.5 }],
        alignmentStatus: 'aligned',
      })),
    };
  }

  async cancel(taskId: string): Promise<ASRCancelResult> {
    this.cancelCalls.push(taskId);
    return this.onCancel ? this.onCancel(taskId) : { taskId, status: 'cancel_requested' };
  }

  async getStatus(): Promise<ASRProviderStatus> {
    this.getStatusCalls += 1;
    return { ...this.status };
  }

  async dispose(): Promise<void> {
    this.disposeCalls += 1;
  }
}

function fixture(id: string, referenceText = '你好'): RunnableSpeechFixture {
  return {
    id,
    audioFileName: id,
    audioPath: `/fixtures/${id}`,
    format: 'wav',
    title: id,
    referenceText,
    error: null,
  };
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

describe('speechSession', () => {
  it('自动初始化使用 2 分钟租约，超时后释放 Provider', async () => {
    const clock = new FakeClock();
    const provider = new FakeProvider();
    const session = new SpeechSession({ providerFactory: () => provider, clock });

    await session.initialize('automatic');

    expect(provider.initializeCalls).toBe(1);
    expect(session.snapshot()).toMatchObject({
      initialized: true,
      leaseMode: 'automatic',
      idleDeadlineMs: SPEECH_LEASE_DURATIONS_MS.automatic,
    });
    clock.advance(SPEECH_LEASE_DURATIONS_MS.automatic - 1);
    expect(provider.disposeCalls).toBe(0);
    clock.advance(1);
    await flushPromises();
    expect(provider.disposeCalls).toBe(1);
    expect(session.snapshot()).toMatchObject({
      initialized: false,
      state: 'uninitialized',
      leaseMode: null,
    });
  });

  it('手动初始化把现有自动租约切为 10 分钟且不重复加载', async () => {
    const clock = new FakeClock();
    const provider = new FakeProvider();
    const session = new SpeechSession({ providerFactory: () => provider, clock });
    await session.initialize('automatic');
    clock.advance(10_000);

    await session.initialize('manual');

    expect(provider.initializeCalls).toBe(1);
    expect(session.snapshot()).toMatchObject({
      leaseMode: 'manual',
      idleDeadlineMs: 10_000 + SPEECH_LEASE_DURATIONS_MS.manual,
    });
  });

  it('sidecar 新进程未初始化时不复用旧缓存并重新初始化', async () => {
    const provider = new FakeProvider();
    const session = new SpeechSession({ providerFactory: () => provider });
    await session.initialize('automatic');
    provider.status = {
      state: 'uninitialized',
      initialized: false,
      currentTaskId: null,
      queuedTaskCount: 0,
      cancelRequested: false,
      loadCount: 0,
    };

    await session.initialize('manual');

    expect(provider.getStatusCalls).toBe(1);
    expect(provider.initializeCalls).toBe(2);
    expect(session.snapshot()).toMatchObject({
      initialized: true,
      state: 'ready',
      leaseMode: 'manual',
    });
  });

  it('转写期间暂停空闲计时，完成后重新开始完整租约', async () => {
    const clock = new FakeClock();
    const provider = new FakeProvider();
    let resolveTranscription: ((value: ASRTranscriptionResult) => void) | undefined;
    provider.onTranscribe = request => new Promise((resolve) => {
      resolveTranscription = resolve;
      expect(request.audioPath).toBe('/fixtures/one.wav');
    });
    const session = new SpeechSession({ providerFactory: () => provider, clock });
    await session.initialize('automatic');

    const run = session.run([fixture('one.wav')], 1);
    await flushPromises();
    expect(session.snapshot()).toMatchObject({ operation: 'transcribing', idleDeadlineMs: null });
    clock.advance(SPEECH_LEASE_DURATIONS_MS.automatic * 2);
    expect(provider.disposeCalls).toBe(0);
    resolveTranscription?.({
      taskId: provider.transcribeCalls[0]!.taskId,
      text: '你好',
      language: 'Chinese',
      timestamps: [],
      alignmentStatus: 'unsupported_language',
    });
    await run;

    expect(session.snapshot().idleDeadlineMs).toBe(
      clock.currentTime + SPEECH_LEASE_DURATIONS_MS.automatic,
    );
  });

  it('手动停用取消当前批次，等待运行结束后释放 Provider', async () => {
    const clock = new FakeClock();
    const provider = new FakeProvider();
    let rejectTranscription: ((error: unknown) => void) | undefined;
    provider.onTranscribe = () => new Promise((_resolve, reject) => {
      rejectTranscription = reject;
    });
    provider.onCancel = async (taskId) => {
      rejectTranscription?.(new ASRProviderError('TASK_CANCELLED', 'cancelled'));
      return { taskId, status: 'cancel_requested' };
    };
    const session = new SpeechSession({ providerFactory: () => provider, clock });
    const run = session.run([fixture('one.wav')], 1);
    await flushPromises();

    const stopped = session.stop();
    await run;
    await stopped;

    expect(provider.cancelCalls).toHaveLength(1);
    expect(provider.disposeCalls).toBe(1);
    expect(session.snapshot()).toMatchObject({ initialized: false, state: 'uninitialized' });
  });

  it('初始化期间停用不会让迟到的初始化结果恢复 Provider', async () => {
    const provider = new FakeProvider();
    let resolveInitialization: ((value: ASRInitializeResult) => void) | undefined;
    provider.onInitialize = () => new Promise((resolve) => {
      resolveInitialization = resolve;
    });
    const session = new SpeechSession({ providerFactory: () => provider });
    const initialization = session.initialize('automatic');
    await flushPromises();

    await session.stop();
    resolveInitialization?.(initializationResult());

    await expect(initialization).rejects.toMatchObject({ code: 'PROVIDER_DISPOSED' });
    expect(session.snapshot()).toMatchObject({ initialized: false, state: 'uninitialized' });
  });

  it('超时释放后再次运行会创建并自动初始化新 Provider', async () => {
    const clock = new FakeClock();
    const providers: FakeProvider[] = [];
    const session = new SpeechSession({
      clock,
      providerFactory: () => {
        const provider = new FakeProvider();
        providers.push(provider);
        return provider;
      },
    });
    await session.initialize('automatic');
    clock.advance(SPEECH_LEASE_DURATIONS_MS.automatic);
    await flushPromises();

    const report = await session.run([fixture('new.wav')], 1);

    expect(providers).toHaveLength(2);
    expect(providers[1]!.initializeCalls).toBe(1);
    expect(report.items[0]!.status).toBe('succeeded');
  });

  it('批量 API 按批次耗时计算行级与整体 CPM', async () => {
    const clock = new FakeClock();
    const provider = new FakeProvider();
    provider.onBatch = async (request) => {
      clock.advance(3_000);
      return {
        taskId: request.taskId,
        items: request.items.map((item, index) => ({
          itemId: item.itemId,
          text: index === 0 ? '你好12' : '测试34',
          language: 'Chinese',
          timestamps: [],
          alignmentStatus: 'unsupported_language',
        })),
      };
    };
    const emit = vi.fn();
    const session = new SpeechSession({ providerFactory: () => provider, clock, emit });

    const report = await session.run([
      fixture('first.wav', '你好12'),
      fixture('second.wav', '测试34'),
    ], 2);

    expect(provider.transcribeCalls).toHaveLength(0);
    expect(provider.batchCalls).toHaveLength(1);
    expect(report).toMatchObject({
      effectiveBatchSize: 2,
      inferenceSeconds: 3,
      successfulCharacterCount: 8,
      overallCpm: 160,
      partial: false,
    });
    expect(report.items.map(item => item.cpmKind)).toEqual(['batch_derived', 'batch_derived']);
    expect(report.items.map(item => item.cpm)).toEqual([80, 80]);
    expect(emit).toHaveBeenCalledWith(expect.objectContaining({ type: 'run-progress' }));
  });

  it('首次自动初始化失败只记录错误，不自行循环重试', async () => {
    const provider = new FakeProvider();
    provider.onInitialize = async () => {
      throw new ASRProviderError('ACCELERATOR_UNAVAILABLE', 'accelerator unavailable');
    };
    const session = new SpeechSession({ providerFactory: () => provider });

    await expect(session.initialize('automatic')).rejects.toMatchObject({ code: 'ACCELERATOR_UNAVAILABLE' });
    await flushPromises();

    expect(provider.initializeCalls).toBe(1);
    expect(session.snapshot()).toMatchObject({
      state: 'failed',
      lastError: { code: 'ACCELERATOR_UNAVAILABLE' },
      idleDeadlineMs: null,
    });
  });
});
