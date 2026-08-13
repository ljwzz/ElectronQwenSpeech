import type {
  ASRBatchProvider,
  TTSInitializeResult,
  TTSProvider,
  TTSProviderState,
  TTSProviderStatus,
  TTSSynthesisResult,
} from '@electron-qwen-speech/application';
import type {
  SpeechClipSummary,
  SpeechDevEvent,
  SpeechDevFailure,
  SpeechFailureStage,
  SpeechGeneratedAudio,
  SpeechLabOperation,
  SpeechLabSnapshot,
  SpeechRoundTripReport,
  SpeechRunReport,
  SpeechRuntimePolicy,
  SpeechSynthesisReport,
  SpeechSynthesisRequest,
  SpeechTranscriptionReport,
} from '../contracts.ts';
import type { RunnableSpeechFixture } from './fixtureCatalog.ts';

import { randomUUID } from 'node:crypto';
import { mkdirSync, mkdtempSync } from 'node:fs';
import { readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { isTTSProviderError, TTSProviderError } from '@electron-qwen-speech/application';
import {
  SPEECH_DEV_RUNTIME_CONFIG,
  SPEECH_DEV_TTS_CONFIG,
} from '../contracts.ts';
import { QwenLocalASRProvider } from './asr/qwenLocalASRProvider.ts';
import { calculateSpeechSimilarity, normalizeSpeechText } from './metrics.ts';
import { SpeechSession, toSpeechFailure } from './speechSession.ts';
import { QwenLocalTTSProvider } from './tts/qwenLocalTTSProvider.ts';

interface InternalClip {
  summary: SpeechClipSummary;
  audioPath: string;
  synthesisWallSeconds: number;
}

export interface SpeechLabSessionOptions {
  asrProviderFactory?: () => ASRBatchProvider;
  ttsProviderFactory?: (outputDirectory: string) => TTSProvider;
  outputDirectory?: string;
  emit?: (event: SpeechDevEvent) => void;
  now?: () => number;
}

class SpeechLabSessionError extends Error {
  readonly code: SpeechDevFailure['code'];

  constructor(code: SpeechDevFailure['code'], message: string) {
    super(message);
    this.name = 'SpeechLabSessionError';
    this.code = code;
  }
}

function cloneFailure(failure: SpeechDevFailure | null): SpeechDevFailure | null {
  return failure ? { ...failure } : null;
}

function stageFailure(error: unknown, stage: SpeechFailureStage): SpeechDevFailure {
  if (isTTSProviderError(error)) {
    return {
      code: error.code,
      message: error.message,
      retryable: error.retryable,
      stage,
    };
  }
  if (error instanceof SpeechLabSessionError) {
    return {
      code: error.code,
      message: error.message,
      retryable: false,
      stage,
    };
  }
  return { ...toSpeechFailure(error), stage };
}

function cloneInitialization(initialization: TTSInitializeResult | null): TTSInitializeResult | null {
  if (!initialization)
    return null;
  return {
    ...initialization,
    supportedLanguages: [...initialization.supportedLanguages],
    supportedVoiceIds: [...initialization.supportedVoiceIds],
  };
}

function requireSynthesisRequest(request: SpeechSynthesisRequest): SpeechSynthesisRequest {
  if (!request.text.trim())
    throw new SpeechLabSessionError('INVALID_REQUEST', 'TTS 文本不能为空。');
  if (!(SPEECH_DEV_TTS_CONFIG.voiceIds as readonly string[]).includes(request.voiceId))
    throw new SpeechLabSessionError('INVALID_REQUEST', `不支持的中文音色：${request.voiceId}`);
  if (request.instruction !== undefined && !request.instruction.trim())
    throw new SpeechLabSessionError('INVALID_REQUEST', '朗读指令不能是空字符串。');
  return {
    text: request.text,
    voiceId: request.voiceId,
    ...(request.instruction === undefined ? {} : { instruction: request.instruction }),
  };
}

export class SpeechLabSession {
  readonly #emit: (event: SpeechDevEvent) => void;
  readonly #now: () => number;
  readonly #ttsProviderFactory: (outputDirectory: string) => TTSProvider;
  readonly #outputDirectory: string;
  readonly #ownsOutputDirectory: boolean;
  readonly #asrSession: SpeechSession;

  #policy: SpeechRuntimePolicy = SPEECH_DEV_RUNTIME_CONFIG.defaultPolicy;
  #operation: SpeechLabOperation = 'idle';
  #activeTaskId: string | null = null;
  #ttsProvider: TTSProvider | undefined;
  #ttsState: TTSProviderState = 'uninitialized';
  #ttsInitialization: TTSInitializeResult | null = null;
  #ttsInitializationSeconds: number | null = null;
  #ttsLastError: SpeechDevFailure | null = null;
  #lastError: SpeechDevFailure | null = null;
  #currentClip: InternalClip | null = null;
  #initializationPromise: Promise<SpeechLabSnapshot> | undefined;
  #runtimePolicyPromise: Promise<void> | undefined;
  #stopPromise: Promise<SpeechLabSnapshot> | undefined;
  #workCompletion: Promise<void> | undefined;
  #resolveWorkCompletion: (() => void) | undefined;
  #stopping = false;
  #runSequence = 0;

  constructor(options: SpeechLabSessionOptions = {}) {
    this.#emit = options.emit ?? (() => undefined);
    this.#now = options.now ?? (() => Date.now());
    this.#ownsOutputDirectory = options.outputDirectory === undefined;
    this.#outputDirectory = options.outputDirectory
      ? path.resolve(options.outputDirectory)
      : mkdtempSync(path.join(os.tmpdir(), 'electron-qwen-speech-lab-'));
    mkdirSync(this.#outputDirectory, { recursive: true });
    this.#ttsProviderFactory = options.ttsProviderFactory
      ?? (outputDirectory => new QwenLocalTTSProvider({ outputDirectory }));
    this.#asrSession = new SpeechSession({
      providerFactory: options.asrProviderFactory ?? (() => new QwenLocalASRProvider()),
      emit: (event) => {
        if (event.type === 'run-progress')
          this.#emit(event);
        else
          this.#emitSnapshot();
      },
    });
  }

  snapshot(): SpeechLabSnapshot {
    const tts = {
      state: this.#ttsState,
      initialized: this.#ttsInitialization !== null,
      activeTaskId: this.#ttsState === 'busy' ? this.#activeTaskId : null,
      initializationSeconds: this.#ttsInitializationSeconds,
      initialization: cloneInitialization(this.#ttsInitialization),
      lastError: cloneFailure(this.#ttsLastError),
    };
    return {
      policy: this.#policy,
      operation: this.#operation,
      activeTaskId: this.#activeTaskId,
      tts,
      asr: this.#asrSession.snapshot(),
      currentClip: this.#currentClip ? { ...this.#currentClip.summary } : null,
      lastError: cloneFailure(this.#lastError),
    };
  }

  async initialize(): Promise<SpeechLabSnapshot> {
    if (this.#initializationPromise)
      return this.#initializationPromise;
    this.#assertIdle();
    this.#operation = 'initializing';
    this.#lastError = null;
    this.#emitSnapshot();
    const initialization = (async (): Promise<SpeechLabSnapshot> => {
      if (this.#policy === 'resident') {
        await this.#initializeTTSWithoutThrowing();
        await this.#initializeASRWithoutThrowing('resident');
      } else {
        await this.#initializeASRWithoutThrowing('manual');
      }
      return this.snapshot();
    })();
    this.#initializationPromise = initialization;
    try {
      await initialization;
    } finally {
      this.#initializationPromise = undefined;
      if (!this.#stopping)
        this.#operation = 'idle';
      this.#emitSnapshot();
    }
    return this.snapshot();
  }

  async setRuntimePolicy(policy: SpeechRuntimePolicy): Promise<SpeechLabSnapshot> {
    if (!SPEECH_DEV_RUNTIME_CONFIG.policies.includes(policy))
      throw new SpeechLabSessionError('INVALID_REQUEST', `未知模型驻留策略：${policy}`);
    this.#assertIdle();
    if (policy === this.#policy)
      return this.snapshot();

    this.#operation = 'switching';
    this.#lastError = null;
    this.#emitSnapshot();
    const transition = (async (): Promise<void> => {
      if (policy === 'exclusive') {
        await this.#disposeTTSWithoutThrowing();
        await this.#stopASRWithoutThrowing();
        this.#policy = policy;
      } else {
        this.#policy = policy;
        await this.#initializeTTSWithoutThrowing();
        await this.#initializeASRWithoutThrowing('resident');
      }
    })();
    this.#runtimePolicyPromise = transition;
    try {
      await transition;
    } finally {
      if (this.#runtimePolicyPromise === transition)
        this.#runtimePolicyPromise = undefined;
      if (!this.#stopping)
        this.#operation = 'idle';
      this.#emitSnapshot();
    }
    return this.snapshot();
  }

  async synthesize(request: SpeechSynthesisRequest): Promise<SpeechSynthesisReport> {
    const normalizedRequest = requireSynthesisRequest(request);
    this.#assertIdle();
    const taskId = `speech-tts-${++this.#runSequence}`;
    const wallStartedAt = this.#now();
    this.#beginWork('synthesizing', taskId);
    try {
      if (this.#policy === 'exclusive')
        await this.#asrSession.stop();
      return await this.#synthesizeInternal(normalizedRequest, taskId, wallStartedAt);
    } catch (error) {
      const failure = stageFailure(error, error instanceof TTSProviderError && error.code === 'INITIALIZATION_FAILED'
        ? 'tts_initialize'
        : 'tts_synthesize');
      this.#lastError = failure;
      throw error;
    } finally {
      this.#finishWork();
    }
  }

  async transcribeCurrentClip(): Promise<SpeechRoundTripReport> {
    const clip = this.#requireCurrentClip();
    this.#assertIdle();
    const runId = `speech-generated-asr-${++this.#runSequence}`;
    const wallStartedAt = this.#now();
    this.#beginWork('transcribing', runId);
    try {
      if (this.#policy === 'exclusive')
        await this.#disposeTTSWithoutThrowing();
      const transcription = await this.#transcribeClipInternal(clip);
      return this.#createRoundTripReport(runId, clip, transcription, wallStartedAt);
    } finally {
      this.#finishWork();
    }
  }

  async runRoundTrip(request: SpeechSynthesisRequest): Promise<SpeechRoundTripReport> {
    const normalizedRequest = requireSynthesisRequest(request);
    this.#assertIdle();
    const runId = `speech-round-trip-${++this.#runSequence}`;
    const wallStartedAt = this.#now();
    this.#beginWork('synthesizing', runId);
    try {
      if (this.#policy === 'exclusive')
        await this.#asrSession.stop();
      await this.#synthesizeInternal(normalizedRequest, `${runId}-tts`, wallStartedAt);
      const clip = this.#requireCurrentClip();
      if (this.#policy === 'exclusive')
        await this.#disposeTTSWithoutThrowing();
      this.#operation = 'transcribing';
      this.#activeTaskId = `${runId}-asr`;
      this.#emitSnapshot();
      const transcription = await this.#transcribeClipInternal(clip);
      return this.#createRoundTripReport(runId, clip, transcription, wallStartedAt);
    } catch (error) {
      const failure = stageFailure(error, 'tts_synthesize');
      this.#lastError = failure;
      throw error;
    } finally {
      this.#finishWork();
    }
  }

  async run(fixtures: RunnableSpeechFixture[], requestedBatchSize: number): Promise<SpeechRunReport> {
    this.#assertIdle();
    const taskId = `speech-fixtures-${++this.#runSequence}`;
    this.#beginWork('transcribing', taskId);
    try {
      if (this.#policy === 'exclusive')
        await this.#disposeTTSWithoutThrowing();
      await this.#asrSession.initialize(this.#policy === 'resident' ? 'resident' : 'manual');
      return await this.#asrSession.run(fixtures, requestedBatchSize);
    } finally {
      this.#finishWork();
    }
  }

  async readGeneratedAudio(clipId: string): Promise<SpeechGeneratedAudio> {
    const clip = this.#currentClip;
    if (!clip || clip.summary.clipId !== clipId)
      throw new SpeechLabSessionError('FORBIDDEN', '生成音频 ID 不存在或已失效。');
    return {
      clipId,
      mimeType: 'audio/wav',
      data: new Uint8Array(await readFile(clip.audioPath)),
    };
  }

  async cancel(): Promise<boolean> {
    const taskId = this.#activeTaskId;
    if (!taskId)
      return false;
    if (this.#operation === 'synthesizing' && this.#ttsProvider) {
      const result = await this.#ttsProvider.cancel(taskId.endsWith('-tts') ? taskId : taskId);
      return result.status !== 'not_found';
    }
    return this.#asrSession.cancel();
  }

  async stop(): Promise<SpeechLabSnapshot> {
    if (this.#stopPromise)
      return this.#stopPromise;
    const stop = this.#performStop();
    this.#stopPromise = stop;
    try {
      return await stop;
    } finally {
      this.#stopPromise = undefined;
    }
  }

  async dispose(): Promise<void> {
    try {
      await this.stop();
    } finally {
      if (this.#ownsOutputDirectory)
        await rm(this.#outputDirectory, { force: true, recursive: true });
    }
  }

  async #performStop(): Promise<SpeechLabSnapshot> {
    const previousOperation = this.#operation;
    const previousTaskId = this.#activeTaskId;
    this.#stopping = true;
    this.#operation = 'stopping';
    this.#emitSnapshot();
    let stopError: unknown;
    try {
      const initialization = this.#initializationPromise;
      const runtimePolicy = this.#runtimePolicyPromise;
      if (initialization) {
        await initialization.catch((error) => {
          stopError = error;
        });
      }
      if (runtimePolicy) {
        await runtimePolicy.catch((error) => {
          stopError ??= error;
        });
      }
      if (previousTaskId && this.#ttsProvider && previousOperation === 'synthesizing') {
        try {
          await this.#ttsProvider.cancel(previousTaskId);
        } catch (error) {
          stopError = error;
        }
      } else {
        await this.#asrSession.cancel().catch((error) => {
          stopError = error;
          return false;
        });
      }
      if (this.#workCompletion)
        await this.#workCompletion;
      await this.#disposeTTSWithoutThrowing();
      await this.#asrSession.stop().catch((error) => {
        stopError ??= error;
      });
    } finally {
      this.#activeTaskId = null;
      this.#operation = 'idle';
      this.#stopping = false;
      if (stopError)
        this.#lastError = stageFailure(stopError, 'runtime_policy');
      this.#emitSnapshot();
    }
    if (stopError)
      throw stopError;
    return this.snapshot();
  }

  async #synthesizeInternal(
    request: SpeechSynthesisRequest,
    taskId: string,
    wallStartedAt: number,
  ): Promise<SpeechSynthesisReport> {
    await this.#ensureTTS();
    const provider = this.#ttsProvider;
    if (!provider)
      throw new TTSProviderError('NOT_INITIALIZED', 'TTS Provider 尚未初始化。');
    this.#ttsState = 'busy';
    this.#activeTaskId = taskId;
    this.#ttsLastError = null;
    this.#emitSnapshot();
    let result: TTSSynthesisResult;
    try {
      result = await provider.synthesize({
        taskId,
        text: request.text,
        language: SPEECH_DEV_TTS_CONFIG.language,
        voiceId: request.voiceId,
        ...(request.instruction === undefined ? {} : { instruction: request.instruction }),
      });
      this.#ttsState = 'ready';
    } catch (error) {
      const failure = stageFailure(error, 'tts_synthesize');
      this.#ttsState = failure.code === 'TASK_CANCELLED' ? 'ready' : 'failed';
      this.#ttsLastError = failure;
      throw error;
    }

    const clipId = randomUUID();
    const summary: SpeechClipSummary = {
      clipId,
      sourceText: request.text,
      language: SPEECH_DEV_TTS_CONFIG.language,
      voiceId: request.voiceId,
      instruction: request.instruction ?? null,
      sampleRate: result.sampleRate,
      channels: result.channels,
      frameCount: result.frameCount,
      durationSeconds: result.durationSeconds,
      generationSeconds: result.generationSeconds,
    };
    const previousClip = this.#currentClip;
    const synthesisWallSeconds = Math.max(0, this.#now() - wallStartedAt) / 1_000;
    this.#currentClip = {
      summary,
      audioPath: result.audioPath,
      synthesisWallSeconds,
    };
    if (previousClip && previousClip.audioPath !== result.audioPath)
      await rm(previousClip.audioPath, { force: true });
    this.#emitSnapshot();
    return { ...summary, wallSeconds: synthesisWallSeconds };
  }

  async #transcribeClipInternal(clip: InternalClip): Promise<SpeechTranscriptionReport> {
    const fixture: RunnableSpeechFixture = {
      id: clip.summary.clipId,
      audioFileName: `${clip.summary.clipId}.wav`,
      audioPath: clip.audioPath,
      format: 'wav',
      title: 'TTS 生成音频',
      referenceText: clip.summary.sourceText,
      error: null,
    };
    try {
      await this.#asrSession.initialize(this.#policy === 'resident' ? 'resident' : 'manual');
      const report = await this.#asrSession.run([fixture], 1);
      const item = report.items[0]!;
      if (item.status !== 'succeeded' || item.transcript === null) {
        const failure = item.error
          ? { ...item.error, stage: 'asr_transcribe' as const }
          : stageFailure(new SpeechLabSessionError('UNKNOWN', 'ASR 未返回成功结果。'), 'asr_transcribe');
        this.#lastError = failure;
        return {
          transcript: item.transcript,
          language: item.language,
          timestamps: item.timestamps.map(timestamp => ({ ...timestamp })),
          alignmentStatus: item.alignmentStatus,
          normalizedMatch: null,
          similarity: item.similarity,
          inferenceSeconds: item.inferenceSeconds,
          failure,
        };
      }
      const normalizedMatch = normalizeSpeechText(item.transcript)
        === normalizeSpeechText(clip.summary.sourceText);
      this.#lastError = null;
      return {
        transcript: item.transcript,
        language: item.language,
        timestamps: item.timestamps.map(timestamp => ({ ...timestamp })),
        alignmentStatus: item.alignmentStatus,
        normalizedMatch,
        similarity: calculateSpeechSimilarity(item.transcript, clip.summary.sourceText),
        inferenceSeconds: item.inferenceSeconds,
        failure: null,
      };
    } catch (error) {
      const failure = stageFailure(error, 'asr_initialize');
      this.#lastError = failure;
      return {
        transcript: null,
        language: null,
        timestamps: [],
        alignmentStatus: null,
        normalizedMatch: null,
        similarity: null,
        inferenceSeconds: null,
        failure,
      };
    }
  }

  #createRoundTripReport(
    runId: string,
    clip: InternalClip,
    transcription: SpeechTranscriptionReport,
    wallStartedAt: number,
  ): SpeechRoundTripReport {
    return {
      runId,
      synthesis: {
        ...clip.summary,
        wallSeconds: clip.synthesisWallSeconds,
      },
      transcription,
      verdict: transcription.failure
        ? 'validation_failed'
        : (transcription.normalizedMatch ? 'consistent' : 'review_required'),
      wallSeconds: Math.max(0, this.#now() - wallStartedAt) / 1_000,
    };
  }

  async #ensureTTS(): Promise<TTSInitializeResult> {
    if (this.#ttsProvider && this.#ttsInitialization) {
      let status: TTSProviderStatus;
      try {
        status = await this.#ttsProvider.getStatus();
      } catch (error) {
        this.#ttsState = 'failed';
        this.#ttsInitialization = null;
        this.#ttsInitializationSeconds = null;
        this.#ttsLastError = stageFailure(error, 'tts_initialize');
        this.#lastError = this.#ttsLastError;
        this.#emitSnapshot();
        throw error;
      }
      if (status.initialized && status.state === 'ready') {
        this.#ttsState = 'ready';
        return this.#ttsInitialization;
      }
      this.#ttsInitialization = null;
      this.#ttsInitializationSeconds = null;
    }
    this.#ttsState = 'initializing';
    this.#ttsLastError = null;
    this.#emitSnapshot();
    const provider = this.#ttsProvider ?? this.#ttsProviderFactory(this.#outputDirectory);
    this.#ttsProvider = provider;
    const startedAt = this.#now();
    try {
      const initialization = await provider.initialize();
      const languages = new Set(initialization.supportedLanguages.map(value => value.toLowerCase()));
      const voices = new Set(initialization.supportedVoiceIds.map(value => value.toLowerCase()));
      if (!languages.has(SPEECH_DEV_TTS_CONFIG.language.toLowerCase())) {
        throw new TTSProviderError('INITIALIZATION_FAILED', 'CustomVoice capability 缺少 Chinese。');
      }
      const missingVoices = SPEECH_DEV_TTS_CONFIG.voiceIds.filter(voice => !voices.has(voice.toLowerCase()));
      if (missingVoices.length > 0) {
        throw new TTSProviderError('INITIALIZATION_FAILED', 'CustomVoice capability 缺少配置音色。', {
          details: { missingVoices },
        });
      }
      this.#ttsInitialization = initialization;
      this.#ttsInitializationSeconds = Math.max(0, this.#now() - startedAt) / 1_000;
      this.#ttsState = 'ready';
      this.#ttsLastError = null;
      return initialization;
    } catch (error) {
      this.#ttsState = 'failed';
      this.#ttsInitialization = null;
      this.#ttsInitializationSeconds = null;
      this.#ttsLastError = stageFailure(error, 'tts_initialize');
      this.#lastError = this.#ttsLastError;
      try {
        await provider.dispose();
      } catch {
        // 初始化错误是主错误，关闭失败保留在 Provider 日志中。
      }
      if (this.#ttsProvider === provider)
        this.#ttsProvider = undefined;
      throw error;
    } finally {
      this.#emitSnapshot();
    }
  }

  async #initializeTTSWithoutThrowing(): Promise<void> {
    try {
      await this.#ensureTTS();
    } catch {
      // snapshot 已记录 TTS 初始化失败，继续初始化 ASR。
    }
  }

  async #initializeASRWithoutThrowing(mode: 'manual' | 'resident'): Promise<void> {
    try {
      await this.#asrSession.initialize(mode);
    } catch (error) {
      this.#lastError = stageFailure(error, 'asr_initialize');
    }
  }

  async #disposeTTSWithoutThrowing(): Promise<void> {
    const provider = this.#ttsProvider;
    if (!provider) {
      this.#ttsState = 'uninitialized';
      this.#ttsInitialization = null;
      this.#ttsInitializationSeconds = null;
      this.#ttsLastError = null;
      return;
    }
    try {
      await provider.dispose();
    } catch (error) {
      this.#lastError = stageFailure(error, 'runtime_policy');
    } finally {
      if (this.#ttsProvider === provider)
        this.#ttsProvider = undefined;
      this.#ttsState = 'uninitialized';
      this.#ttsInitialization = null;
      this.#ttsInitializationSeconds = null;
      this.#ttsLastError = null;
      this.#emitSnapshot();
    }
  }

  async #stopASRWithoutThrowing(): Promise<void> {
    try {
      await this.#asrSession.stop();
    } catch (error) {
      this.#lastError = stageFailure(error, 'runtime_policy');
    }
  }

  #beginWork(operation: Extract<SpeechLabOperation, 'synthesizing' | 'transcribing'>, taskId: string): void {
    this.#operation = operation;
    this.#activeTaskId = taskId;
    this.#lastError = null;
    this.#workCompletion = new Promise<void>((resolve) => {
      this.#resolveWorkCompletion = resolve;
    });
    this.#emitSnapshot();
  }

  #finishWork(): void {
    this.#activeTaskId = null;
    this.#resolveWorkCompletion?.();
    this.#resolveWorkCompletion = undefined;
    this.#workCompletion = undefined;
    if (!this.#stopping)
      this.#operation = 'idle';
    this.#emitSnapshot();
  }

  #assertIdle(): void {
    if (this.#operation !== 'idle' || this.#stopping)
      throw new SpeechLabSessionError('SESSION_BUSY', '语音诊断台已有任务正在运行。');
  }

  #requireCurrentClip(): InternalClip {
    if (!this.#currentClip)
      throw new SpeechLabSessionError('INVALID_REQUEST', '尚未生成可转写的 TTS 音频。');
    return this.#currentClip;
  }

  #emitSnapshot(): void {
    this.#emit({ type: 'snapshot', snapshot: this.snapshot() });
  }
}
