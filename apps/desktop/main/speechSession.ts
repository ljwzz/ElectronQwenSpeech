import type {
  ASRBatchProvider,
  ASRInitializeResult,
  ASRTranscriptionResult,
} from '@electron-qwen-speech/application';
import type {
  SpeechDevFailure,
  SpeechLeaseMode,
  SpeechProviderSnapshot,
  SpeechRunItem,
  SpeechRunProgressEvent,
  SpeechRunReport,
} from '../contracts.ts';
import type { RunnableSpeechFixture } from './fixtureCatalog.ts';

import { ASRProviderError, isASRProviderError } from '@electron-qwen-speech/application';
import { QwenLocalASRProvider } from './asr/qwenLocalASRProvider.ts';
import { calculateCpm, calculateSpeechSimilarity, countSpeechCharacters } from './metrics.ts';

export const SPEECH_LEASE_DURATIONS_MS = {
  automatic: 2 * 60 * 1_000,
  manual: 10 * 60 * 1_000,
} as const;

interface TimerClock {
  now: () => number;
  setTimeout: (handler: () => void, delayMs: number) => unknown;
  clearTimeout: (handle: unknown) => void;
}

export interface SpeechSessionOptions {
  providerFactory?: () => ASRBatchProvider;
  clock?: TimerClock;
  emit?: (event: SpeechSessionEvent) => void;
}

export type SpeechSessionEvent = SpeechRunProgressEvent | {
  type: 'snapshot';
  snapshot: SpeechProviderSnapshot;
};

class SpeechSessionError extends Error {
  readonly code: SpeechDevFailure['code'];

  constructor(code: SpeechDevFailure['code'], message: string) {
    super(message);
    this.name = 'SpeechSessionError';
    this.code = code;
  }
}

const DEFAULT_CLOCK: TimerClock = {
  now: () => Date.now(),
  setTimeout: (handler, delayMs) => setTimeout(handler, delayMs),
  clearTimeout: handle => clearTimeout(handle as NodeJS.Timeout),
};

export function toSpeechFailure(error: unknown): SpeechDevFailure {
  if (isASRProviderError(error)) {
    return {
      code: error.code,
      message: error.message,
      retryable: error.retryable,
    };
  }
  if (error instanceof SpeechSessionError) {
    return {
      code: error.code,
      message: error.message,
      retryable: false,
    };
  }
  return {
    code: 'UNKNOWN',
    message: error instanceof Error ? error.message : '未知错误。',
    retryable: false,
  };
}

function createPendingItem(fixture: RunnableSpeechFixture): SpeechRunItem {
  return {
    fixtureId: fixture.id,
    audioFileName: fixture.audioFileName,
    title: fixture.title,
    referenceText: fixture.referenceText,
    status: 'pending',
    transcript: null,
    language: null,
    timestamps: [],
    similarity: null,
    characterCount: 0,
    inferenceSeconds: null,
    cpm: null,
    cpmKind: 'exact',
    error: null,
  };
}

export class SpeechSession {
  readonly #providerFactory: () => ASRBatchProvider;
  readonly #clock: TimerClock;
  readonly #emit: (event: SpeechSessionEvent) => void;

  #provider: ASRBatchProvider | undefined;
  #state: SpeechProviderSnapshot['state'] = 'uninitialized';
  #operation: SpeechProviderSnapshot['operation'] = 'idle';
  #initialization: ASRInitializeResult | null = null;
  #initializationSeconds: number | null = null;
  #leaseMode: SpeechLeaseMode = null;
  #idleDeadlineMs: number | null = null;
  #idleTimer: unknown;
  #activeTaskId: string | null = null;
  #lastError: SpeechDevFailure | null = null;
  #initializationPromise: Promise<SpeechProviderSnapshot> | undefined;
  #runCompletion: Promise<void> | undefined;
  #resolveRunCompletion: (() => void) | undefined;
  #stopPromise: Promise<SpeechProviderSnapshot> | undefined;
  #cancelRequested = false;
  #runSequence = 0;
  #stopping = false;

  constructor(options: SpeechSessionOptions = {}) {
    this.#providerFactory = options.providerFactory ?? (() => new QwenLocalASRProvider());
    this.#clock = options.clock ?? DEFAULT_CLOCK;
    this.#emit = options.emit ?? (() => undefined);
  }

  snapshot(): SpeechProviderSnapshot {
    return {
      state: this.#state,
      operation: this.#operation,
      initialized: this.#initialization !== null,
      activeTaskId: this.#activeTaskId,
      leaseMode: this.#leaseMode,
      idleDeadlineMs: this.#idleDeadlineMs,
      initializationSeconds: this.#initializationSeconds,
      initialization: this.#initialization ? { ...this.#initialization } : null,
      lastError: this.#lastError ? { ...this.#lastError } : null,
    };
  }

  async initialize(mode: Exclude<SpeechLeaseMode, null>): Promise<SpeechProviderSnapshot> {
    this.#setLeaseMode(mode);
    if (this.#initialization && this.#provider) {
      if (this.#operation === 'idle')
        this.#scheduleIdleStop();
      this.#emitSnapshot();
      return this.snapshot();
    }
    if (this.#initializationPromise)
      return this.#initializationPromise;
    if (this.#stopping && this.#stopPromise)
      await this.#stopPromise;

    this.#clearIdleTimer();
    this.#operation = 'initializing';
    this.#state = 'initializing';
    this.#lastError = null;
    this.#provider ??= this.#providerFactory();
    const provider = this.#provider;
    const startedAt = this.#clock.now();
    this.#emitSnapshot();

    const initializationPromise = (async (): Promise<SpeechProviderSnapshot> => {
      try {
        const initialization = await provider.initialize();
        if (this.#provider !== provider || this.#stopping)
          throw new ASRProviderError('PROVIDER_DISPOSED', '初始化期间 Provider 已停用。');
        this.#initialization = initialization;
        this.#initializationSeconds = Math.max(0, this.#clock.now() - startedAt) / 1_000;
        this.#state = 'ready';
        this.#lastError = null;
        return this.snapshot();
      } catch (error) {
        if (!this.#stopping && this.#provider === provider) {
          this.#initialization = null;
          this.#initializationSeconds = null;
          this.#state = 'failed';
          this.#lastError = toSpeechFailure(error);
        }
        throw error;
      } finally {
        this.#initializationPromise = undefined;
        if (!this.#stopping) {
          this.#operation = 'idle';
          if (this.#initialization)
            this.#scheduleIdleStop();
        }
        this.#emitSnapshot();
      }
    })();
    this.#initializationPromise = initializationPromise;
    return initializationPromise;
  }

  async run(fixtures: RunnableSpeechFixture[], requestedBatchSize: number): Promise<SpeechRunReport> {
    if (!Number.isSafeInteger(requestedBatchSize) || requestedBatchSize <= 0)
      throw new ASRProviderError('INVALID_REQUEST', 'batchSize 必须是正安全整数。');
    if (fixtures.length === 0)
      throw new ASRProviderError('INVALID_REQUEST', '至少需要一个 fixture。');
    if (this.#runCompletion)
      throw new SpeechSessionError('SESSION_BUSY', '已有转写任务正在运行。');

    await this.initialize('automatic');
    const provider = this.#provider;
    if (!provider || !this.#initialization)
      throw new ASRProviderError('NOT_INITIALIZED', 'ASR Provider 尚未初始化。');

    this.#clearIdleTimer();
    this.#operation = 'transcribing';
    this.#state = 'busy';
    this.#cancelRequested = false;
    this.#lastError = null;
    let resolveCompletion: (() => void) | undefined;
    this.#runCompletion = new Promise<void>((resolve) => {
      resolveCompletion = resolve;
    });
    this.#resolveRunCompletion = resolveCompletion;

    const runId = `speech-run-${++this.#runSequence}`;
    const effectiveBatchSize = Math.min(requestedBatchSize, fixtures.length);
    const items = fixtures.map(createPendingItem);
    const wallStartedAt = this.#clock.now();
    let inferenceSeconds = 0;
    this.#emitProgress(runId, items);
    this.#emitSnapshot();

    try {
      for (let offset = 0; offset < fixtures.length; offset += effectiveBatchSize) {
        if (this.#cancelRequested) {
          this.#markRemainingCancelled(items, offset);
          break;
        }
        const fixtureGroup = fixtures.slice(offset, offset + effectiveBatchSize);
        const itemGroup = items.slice(offset, offset + effectiveBatchSize);
        const taskId = `${runId}-batch-${Math.floor(offset / effectiveBatchSize) + 1}`;
        this.#activeTaskId = taskId;
        for (const item of itemGroup)
          item.status = 'running';
        this.#emitProgress(runId, items);
        this.#emitSnapshot();

        const inferenceStartedAt = this.#clock.now();
        try {
          if (fixtureGroup.length === 1) {
            const fixture = fixtureGroup[0]!;
            const result = await provider.transcribe({
              taskId,
              audioPath: fixture.audioPath,
            });
            const elapsedSeconds = Math.max(0, this.#clock.now() - inferenceStartedAt) / 1_000;
            inferenceSeconds += elapsedSeconds;
            this.#applyResult(itemGroup[0]!, result, elapsedSeconds, 'exact');
          } else {
            const result = await provider.transcribeBatch({
              taskId,
              items: fixtureGroup.map(fixture => ({
                itemId: fixture.id,
                audioPath: fixture.audioPath,
              })),
            });
            const elapsedSeconds = Math.max(0, this.#clock.now() - inferenceStartedAt) / 1_000;
            inferenceSeconds += elapsedSeconds;
            for (const [index, batchResult] of result.items.entries()) {
              this.#applyResult(itemGroup[index]!, {
                taskId,
                text: batchResult.text,
                ...(batchResult.language === undefined ? {} : { language: batchResult.language }),
                timestamps: batchResult.timestamps,
              }, elapsedSeconds, 'batch_derived');
            }
          }
        } catch (error) {
          const elapsedSeconds = Math.max(0, this.#clock.now() - inferenceStartedAt) / 1_000;
          inferenceSeconds += elapsedSeconds;
          const failure = toSpeechFailure(error);
          const cancelled = failure.code === 'TASK_CANCELLED' || this.#cancelRequested;
          for (const item of itemGroup) {
            item.status = cancelled ? 'cancelled' : 'failed';
            item.inferenceSeconds = elapsedSeconds;
            item.error = failure;
            item.cpmKind = itemGroup.length === 1 ? 'exact' : 'batch_derived';
          }
          this.#lastError = failure;
          if (cancelled) {
            this.#markRemainingCancelled(items, offset + fixtureGroup.length);
            break;
          }
        } finally {
          this.#activeTaskId = null;
          this.#emitProgress(runId, items);
          this.#emitSnapshot();
        }
      }

      const successfulItems = items.filter(item => item.status === 'succeeded');
      const successfulCharacterCount = successfulItems.reduce(
        (total, item) => total + item.characterCount,
        0,
      );
      return {
        runId,
        requestedBatchSize,
        effectiveBatchSize,
        items,
        wallSeconds: Math.max(0, this.#clock.now() - wallStartedAt) / 1_000,
        inferenceSeconds,
        successfulCharacterCount,
        overallCpm: calculateCpm(successfulCharacterCount, inferenceSeconds),
        partial: successfulItems.length !== items.length,
      };
    } finally {
      this.#activeTaskId = null;
      this.#runCompletion = undefined;
      this.#resolveRunCompletion?.();
      this.#resolveRunCompletion = undefined;
      if (!this.#stopping) {
        this.#operation = 'idle';
        this.#state = this.#initialization ? 'ready' : 'uninitialized';
        this.#scheduleIdleStop();
      }
      this.#emitSnapshot();
    }
  }

  async cancel(): Promise<boolean> {
    const provider = this.#provider;
    const taskId = this.#activeTaskId;
    if (!provider || !taskId)
      return false;
    this.#cancelRequested = true;
    const result = await provider.cancel(taskId);
    return result.status !== 'not_found';
  }

  async stop(): Promise<SpeechProviderSnapshot> {
    if (this.#stopPromise)
      return this.#stopPromise;
    const stopPromise = this.#performStop();
    this.#stopPromise = stopPromise;
    try {
      return await stopPromise;
    } finally {
      this.#stopPromise = undefined;
    }
  }

  async #performStop(): Promise<SpeechProviderSnapshot> {
    this.#stopping = true;
    this.#clearIdleTimer();
    this.#operation = 'stopping';
    this.#state = 'shutting_down';
    this.#emitSnapshot();
    const provider = this.#provider;
    let stopError: unknown;
    try {
      if (provider && this.#activeTaskId) {
        this.#cancelRequested = true;
        try {
          await provider.cancel(this.#activeTaskId);
        } catch (error) {
          stopError = error;
        }
      }
      if (this.#runCompletion)
        await this.#runCompletion;
      if (provider) {
        try {
          await provider.dispose();
        } catch (error) {
          stopError ??= error;
        }
      }
    } finally {
      this.#provider = undefined;
      this.#initialization = null;
      this.#initializationSeconds = null;
      this.#leaseMode = null;
      this.#idleDeadlineMs = null;
      this.#activeTaskId = null;
      this.#operation = 'idle';
      this.#state = 'uninitialized';
      this.#stopping = false;
      if (stopError)
        this.#lastError = toSpeechFailure(stopError);
      else
        this.#lastError = null;
      this.#emitSnapshot();
    }
    if (stopError)
      throw stopError;
    return this.snapshot();
  }

  #setLeaseMode(mode: Exclude<SpeechLeaseMode, null>): void {
    if (mode === 'resident' || this.#leaseMode === null)
      this.#leaseMode = mode;
    else if (mode === 'manual' && this.#leaseMode !== 'resident')
      this.#leaseMode = 'manual';
  }

  #scheduleIdleStop(): void {
    this.#clearIdleTimer();
    if (
      !this.#initialization
      || !this.#leaseMode
      || this.#leaseMode === 'resident'
      || this.#operation !== 'idle'
      || this.#stopping
    ) {
      return;
    }
    const delayMs = SPEECH_LEASE_DURATIONS_MS[this.#leaseMode];
    this.#idleDeadlineMs = this.#clock.now() + delayMs;
    this.#idleTimer = this.#clock.setTimeout(() => {
      this.#idleTimer = undefined;
      this.#idleDeadlineMs = null;
      void this.stop().catch(() => undefined);
    }, delayMs);
  }

  #clearIdleTimer(): void {
    if (this.#idleTimer !== undefined)
      this.#clock.clearTimeout(this.#idleTimer);
    this.#idleTimer = undefined;
    this.#idleDeadlineMs = null;
  }

  #applyResult(
    item: SpeechRunItem,
    result: ASRTranscriptionResult,
    inferenceSeconds: number,
    cpmKind: SpeechRunItem['cpmKind'],
  ): void {
    const characterCount = countSpeechCharacters(result.text);
    item.status = 'succeeded';
    item.transcript = result.text;
    item.language = result.language ?? null;
    item.timestamps = result.timestamps;
    item.similarity = calculateSpeechSimilarity(result.text, item.referenceText);
    item.characterCount = characterCount;
    item.inferenceSeconds = inferenceSeconds;
    item.cpm = calculateCpm(characterCount, inferenceSeconds);
    item.cpmKind = cpmKind;
    item.error = null;
  }

  #markRemainingCancelled(items: SpeechRunItem[], offset: number): void {
    const cancellation = toSpeechFailure(new ASRProviderError('TASK_CANCELLED', '运行已取消。'));
    for (const item of items.slice(offset)) {
      if (item.status === 'pending') {
        item.status = 'cancelled';
        item.error = cancellation;
      }
    }
  }

  #emitSnapshot(): void {
    this.#emit({ type: 'snapshot', snapshot: this.snapshot() });
  }

  #emitProgress(runId: string, items: SpeechRunItem[]): void {
    this.#emit({
      type: 'run-progress',
      runId,
      completed: items.filter(item => item.status !== 'pending' && item.status !== 'running').length,
      total: items.length,
      items: items.map(item => ({
        ...item,
        timestamps: item.timestamps.map(timestamp => ({ ...timestamp })),
        error: item.error ? { ...item.error } : null,
      })),
    });
  }
}
