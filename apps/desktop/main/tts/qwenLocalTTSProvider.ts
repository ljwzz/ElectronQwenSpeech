import type {
  TTSCancelResult,
  TTSInitializeResult,
  TTSProvider,
  TTSProviderErrorCode,
  TTSProviderHealth,
  TTSProviderState,
  TTSProviderStatus,
  TTSSynthesisRequest,
  TTSSynthesisResult,
} from '@electron-qwen-speech/application';
import type { Buffer } from 'node:buffer';
import type { spawn } from 'node:child_process';

import { randomUUID } from 'node:crypto';
import { mkdirSync, mkdtempSync } from 'node:fs';
import { readFile, realpath, rm, stat } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { TTS_PROVIDER_ERROR_CODES, TTSProviderError } from '@electron-qwen-speech/application';
import { findDevelopmentRepositoryRoot } from '../developmentRepositoryRoot.ts';
import { JsonLineSidecarClient, JsonLineSidecarError } from '../sidecar/jsonLineSidecarClient.ts';

const REPOSITORY_ROOT = findDevelopmentRepositoryRoot();
const TTS_MODEL_PATH_ENV = 'ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH';
const TTS_ERROR_CODE_SET = new Set<string>(TTS_PROVIDER_ERROR_CODES);
const TTS_PROVIDER_STATE_SET = new Set<TTSProviderState>([
  'uninitialized',
  'initializing',
  'ready',
  'busy',
  'failed',
  'shutting_down',
  'disposed',
]);

export const QWEN_LOCAL_TTS_DEFAULTS = {
  pythonExecutable: path.join(REPOSITORY_ROOT, 'services', 'tts-sidecar', '.venv', 'bin', 'python'),
  sidecarEntryPoint: path.join(REPOSITORY_ROOT, 'services', 'tts-sidecar', 'main.py'),
  workingDirectory: REPOSITORY_ROOT,
  timeouts: {
    controlMs: 5_000,
    initializeMs: 300_000,
    synthesizeMs: 300_000,
    shutdownMs: 15_000,
  },
} as const;

export interface QwenLocalTTSTimeouts {
  controlMs: number;
  initializeMs: number;
  synthesizeMs: number;
  shutdownMs: number;
}

export interface QwenLocalTTSProviderOptions {
  pythonExecutable?: string;
  sidecarEntryPoint?: string;
  workingDirectory?: string;
  outputDirectory?: string;
  timeouts?: Partial<QwenLocalTTSTimeouts>;
  environment?: NodeJS.ProcessEnv;
  spawnProcess?: typeof spawn;
  onStderr?: (output: string) => void;
}

interface ParsedWaveMetadata {
  sampleRate: number;
  channels: number;
  frameCount: number;
  durationSeconds: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function protocolError(message: string, details?: unknown): TTSProviderError {
  return new TTSProviderError('SIDECAR_PROTOCOL_ERROR', message, { details });
}

function readString(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== 'string')
    throw protocolError(`TTS Sidecar 响应字段 ${key} 必须是字符串。`, record);
  return value;
}

function readBoolean(record: Record<string, unknown>, key: string): boolean {
  const value = record[key];
  if (typeof value !== 'boolean')
    throw protocolError(`TTS Sidecar 响应字段 ${key} 必须是布尔值。`, record);
  return value;
}

function readNonNegativeInteger(record: Record<string, unknown>, key: string): number {
  const value = record[key];
  if (!Number.isInteger(value) || (value as number) < 0)
    throw protocolError(`TTS Sidecar 响应字段 ${key} 必须是非负整数。`, record);
  return value as number;
}

function readPositiveInteger(record: Record<string, unknown>, key: string): number {
  const value = readNonNegativeInteger(record, key);
  if (value <= 0)
    throw protocolError(`TTS Sidecar 响应字段 ${key} 必须大于 0。`, record);
  return value;
}

function readNonNegativeNumber(record: Record<string, unknown>, key: string): number {
  const value = record[key];
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0)
    throw protocolError(`TTS Sidecar 响应字段 ${key} 必须是非负有限数。`, record);
  return value;
}

function readStringArray(record: Record<string, unknown>, key: string): string[] {
  const value = record[key];
  if (!Array.isArray(value) || !value.every(item => typeof item === 'string' && item.length > 0))
    throw protocolError(`TTS Sidecar 响应字段 ${key} 必须是非空字符串数组。`, record);
  return [...value];
}

function parseStatus(value: unknown): TTSProviderStatus {
  if (!isRecord(value))
    throw protocolError('TTS Sidecar 状态响应必须是对象。', value);
  const state = readString(value, 'state') as TTSProviderState;
  if (!TTS_PROVIDER_STATE_SET.has(state))
    throw protocolError(`TTS Sidecar 返回未知状态：${state}`, value);
  const currentTaskId = value.currentTaskId;
  if (currentTaskId !== null && typeof currentTaskId !== 'string')
    throw protocolError('TTS Sidecar currentTaskId 必须是字符串或 null。', value);
  return {
    state,
    initialized: readBoolean(value, 'initialized'),
    currentTaskId,
    queuedTaskCount: readNonNegativeInteger(value, 'queuedTaskCount'),
    cancelRequested: readBoolean(value, 'cancelRequested'),
    loadCount: readNonNegativeInteger(value, 'loadCount'),
  };
}

function parseHealth(value: unknown): TTSProviderHealth {
  if (!isRecord(value))
    throw protocolError('TTS Sidecar 健康响应必须是对象。', value);
  return { ...parseStatus(value), healthy: readBoolean(value, 'healthy') };
}

function parseInitializeResult(value: unknown): TTSInitializeResult {
  if (!isRecord(value) || value.state !== 'ready')
    throw protocolError('TTS Sidecar 初始化完成后必须返回 ready 对象。', value);
  return {
    state: 'ready',
    modelPath: readString(value, 'modelPath'),
    device: readString(value, 'device'),
    dtype: readString(value, 'dtype'),
    supportedLanguages: readStringArray(value, 'supportedLanguages'),
    supportedVoiceIds: readStringArray(value, 'supportedVoiceIds'),
    loadCount: readNonNegativeInteger(value, 'loadCount'),
  };
}

function parseSynthesisResult(value: unknown): TTSSynthesisResult {
  if (!isRecord(value))
    throw protocolError('TTS Sidecar 合成响应必须是对象。', value);
  if (value.mimeType !== 'audio/wav')
    throw protocolError('TTS Sidecar 只允许返回 audio/wav。', value);
  const durationSeconds = readNonNegativeNumber(value, 'durationSeconds');
  if (durationSeconds <= 0)
    throw protocolError('TTS Sidecar durationSeconds 必须大于 0。', value);
  return {
    taskId: readString(value, 'taskId'),
    audioPath: readString(value, 'audioPath'),
    mimeType: 'audio/wav',
    sampleRate: readPositiveInteger(value, 'sampleRate'),
    channels: readPositiveInteger(value, 'channels'),
    frameCount: readPositiveInteger(value, 'frameCount'),
    durationSeconds,
    generationSeconds: readNonNegativeNumber(value, 'generationSeconds'),
  };
}

function parseCancelResult(value: unknown): TTSCancelResult {
  if (!isRecord(value))
    throw protocolError('TTS Sidecar 取消响应必须是对象。', value);
  const status = readString(value, 'status');
  if (status !== 'cancelled' && status !== 'cancel_requested' && status !== 'not_found')
    throw protocolError(`TTS Sidecar 返回未知取消状态：${status}`, value);
  return { taskId: readString(value, 'taskId'), status };
}

function normalizeTimeout(value: number | undefined, fallback: number, name: string): number {
  const normalized = value ?? fallback;
  if (!Number.isInteger(normalized) || normalized <= 0)
    throw new TTSProviderError('INVALID_REQUEST', `${name} 必须是正整数毫秒值。`);
  return normalized;
}

function disposedStatus(): TTSProviderStatus {
  return {
    state: 'disposed',
    initialized: false,
    currentTaskId: null,
    queuedTaskCount: 0,
    cancelRequested: false,
    loadCount: 0,
  };
}

function parsePcmWave(buffer: Buffer): ParsedWaveMetadata {
  if (buffer.length < 44 || buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE')
    throw new TTSProviderError('AUDIO_OUTPUT_INVALID', 'TTS 输出不是有效的 RIFF/WAVE 文件。');

  let offset = 12;
  let sampleRate: number | undefined;
  let channels: number | undefined;
  let bytesPerFrame: number | undefined;
  let dataBytes: number | undefined;
  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkStart = offset + 8;
    const chunkEnd = chunkStart + chunkSize;
    if (chunkEnd > buffer.length)
      throw new TTSProviderError('AUDIO_OUTPUT_INVALID', 'TTS WAV chunk 超出文件边界。');
    if (chunkId === 'fmt ') {
      if (chunkSize < 16)
        throw new TTSProviderError('AUDIO_OUTPUT_INVALID', 'TTS WAV fmt chunk 长度不足。');
      const format = buffer.readUInt16LE(chunkStart);
      channels = buffer.readUInt16LE(chunkStart + 2);
      sampleRate = buffer.readUInt32LE(chunkStart + 4);
      const bitsPerSample = buffer.readUInt16LE(chunkStart + 14);
      if (format !== 1 || bitsPerSample !== 16)
        throw new TTSProviderError('AUDIO_OUTPUT_INVALID', 'TTS WAV 必须是 PCM 16-bit。');
      bytesPerFrame = channels * bitsPerSample / 8;
    } else if (chunkId === 'data') {
      dataBytes = chunkSize;
    }
    offset = chunkEnd + (chunkSize % 2);
  }
  if (!sampleRate || !channels || !bytesPerFrame || !dataBytes || dataBytes % bytesPerFrame !== 0)
    throw new TTSProviderError('AUDIO_OUTPUT_INVALID', 'TTS WAV 缺少有效 fmt 或 data chunk。');
  const frameCount = dataBytes / bytesPerFrame;
  return {
    sampleRate,
    channels,
    frameCount,
    durationSeconds: frameCount / sampleRate,
  };
}

function mapSidecarError(error: unknown): TTSProviderError {
  if (error instanceof TTSProviderError)
    return error;
  if (!(error instanceof JsonLineSidecarError))
    return new TTSProviderError('SIDECAR_UNAVAILABLE', 'TTS Sidecar 请求失败。', { cause: error });
  if (error.code === 'REMOTE_ERROR') {
    if (!error.remoteCode || !TTS_ERROR_CODE_SET.has(error.remoteCode)) {
      return protocolError(`TTS Sidecar 返回未知错误码：${error.remoteCode ?? 'undefined'}`, {
        details: error.details,
      });
    }
    return new TTSProviderError(error.remoteCode as TTSProviderErrorCode, error.message, {
      retryable: error.retryable,
      details: error.details,
    });
  }
  const codeByTransportError: Record<Exclude<JsonLineSidecarError['code'], 'REMOTE_ERROR'>, TTSProviderErrorCode> = {
    DISPOSED: 'PROVIDER_DISPOSED',
    EXITED: 'SIDECAR_EXITED',
    PROTOCOL_ERROR: 'SIDECAR_PROTOCOL_ERROR',
    SHUTDOWN_TIMEOUT: 'SHUTDOWN_TIMEOUT',
    TIMEOUT: 'REQUEST_TIMEOUT',
    UNAVAILABLE: 'SIDECAR_UNAVAILABLE',
  };
  return new TTSProviderError(codeByTransportError[error.code], error.message, {
    retryable: error.retryable,
    details: error.details,
    cause: error,
  });
}

export class QwenLocalTTSProvider implements TTSProvider {
  readonly #timeouts: QwenLocalTTSTimeouts;
  readonly #outputDirectory: string;
  readonly #ownsOutputDirectory: boolean;
  readonly #client: JsonLineSidecarClient;
  readonly #modelPath: string;
  #disposed = false;

  constructor(options: QwenLocalTTSProviderOptions = {}) {
    const environment = {
      ...process.env,
      ...options.environment,
    };
    const configuredModelPath = environment[TTS_MODEL_PATH_ENV]?.trim();
    if (!configuredModelPath || !path.isAbsolute(configuredModelPath)) {
      throw new TTSProviderError(
        'INITIALIZATION_FAILED',
        `${TTS_MODEL_PATH_ENV} 必须配置为绝对模型目录。`,
        { details: { environmentVariable: TTS_MODEL_PATH_ENV } },
      );
    }
    this.#modelPath = path.resolve(configuredModelPath);
    this.#timeouts = {
      controlMs: normalizeTimeout(options.timeouts?.controlMs, QWEN_LOCAL_TTS_DEFAULTS.timeouts.controlMs, 'controlMs'),
      initializeMs: normalizeTimeout(options.timeouts?.initializeMs, QWEN_LOCAL_TTS_DEFAULTS.timeouts.initializeMs, 'initializeMs'),
      synthesizeMs: normalizeTimeout(options.timeouts?.synthesizeMs, QWEN_LOCAL_TTS_DEFAULTS.timeouts.synthesizeMs, 'synthesizeMs'),
      shutdownMs: normalizeTimeout(options.timeouts?.shutdownMs, QWEN_LOCAL_TTS_DEFAULTS.timeouts.shutdownMs, 'shutdownMs'),
    };
    this.#ownsOutputDirectory = options.outputDirectory === undefined;
    this.#outputDirectory = options.outputDirectory
      ? path.resolve(options.outputDirectory)
      : mkdtempSync(path.join(os.tmpdir(), 'electron-qwen-speech-tts-'));
    mkdirSync(this.#outputDirectory, { recursive: true });
    this.#client = new JsonLineSidecarClient({
      label: 'TTS Sidecar',
      executable: options.pythonExecutable ?? QWEN_LOCAL_TTS_DEFAULTS.pythonExecutable,
      arguments: ['-u', options.sidecarEntryPoint ?? QWEN_LOCAL_TTS_DEFAULTS.sidecarEntryPoint],
      workingDirectory: options.workingDirectory ?? QWEN_LOCAL_TTS_DEFAULTS.workingDirectory,
      environment: {
        ...environment,
        PYTHONUNBUFFERED: '1',
        PYTORCH_ENABLE_MPS_FALLBACK: '0',
        [TTS_MODEL_PATH_ENV]: this.#modelPath,
        ELECTRON_QWEN_SPEECH_TTS_OUTPUT_ROOT: this.#outputDirectory,
      },
      shutdownTimeoutMs: this.#timeouts.shutdownMs,
      ...(options.spawnProcess === undefined ? {} : { spawnProcess: options.spawnProcess }),
      ...(options.onStderr === undefined ? {} : { onStderr: options.onStderr }),
    });
  }

  async initialize(): Promise<TTSInitializeResult> {
    const result = await this.#call(() => this.#client.request(
      'initialize',
      {},
      this.#timeouts.initializeMs,
    ).then(parseInitializeResult));
    if (path.resolve(result.modelPath) !== this.#modelPath) {
      throw new TTSProviderError('INITIALIZATION_FAILED', 'TTS Sidecar 返回的模型路径与配置不一致。', {
        details: { expected: this.#modelPath, actual: result.modelPath },
      });
    }
    return result;
  }

  async health(): Promise<TTSProviderHealth> {
    return this.#call(() => this.#client.request('health', {}, this.#timeouts.controlMs).then(parseHealth));
  }

  async synthesize(request: TTSSynthesisRequest): Promise<TTSSynthesisResult> {
    if (!request.taskId.trim())
      throw new TTSProviderError('INVALID_REQUEST', 'taskId 不能为空。');
    if (!request.text.trim())
      throw new TTSProviderError('INVALID_REQUEST', 'text 不能为空。');
    if (!request.language.trim())
      throw new TTSProviderError('INVALID_REQUEST', 'language 不能为空。');
    if (!request.voiceId.trim())
      throw new TTSProviderError('INVALID_REQUEST', 'voiceId 不能为空。');
    if (request.instruction !== undefined && !request.instruction.trim())
      throw new TTSProviderError('INVALID_REQUEST', 'instruction 不能是空字符串。');

    const audioPath = path.join(this.#outputDirectory, `${randomUUID()}.wav`);
    const params: Record<string, unknown> = {
      taskId: request.taskId,
      text: request.text,
      language: request.language,
      voiceId: request.voiceId,
      outputPath: audioPath,
    };
    if (request.instruction !== undefined)
      params.instruction = request.instruction;

    try {
      const result = await this.#call(() => this.#client.request(
        'synthesize',
        params,
        this.#timeouts.synthesizeMs,
      ).then(parseSynthesisResult));
      if (result.taskId !== request.taskId) {
        throw protocolError('TTS Sidecar 合成响应与请求不一致。', {
          expectedTaskId: request.taskId,
          actualTaskId: result.taskId,
          expectedAudioPath: audioPath,
          actualAudioPath: result.audioPath,
        });
      }
      const [actualRealPath, expectedRealPath] = await Promise.all([
        realpath(result.audioPath),
        realpath(audioPath),
      ]);
      if (actualRealPath !== expectedRealPath) {
        throw protocolError('TTS Sidecar 合成响应路径与请求不一致。', {
          expectedAudioPath: expectedRealPath,
          actualAudioPath: actualRealPath,
        });
      }
      const fileStats = await stat(audioPath);
      if (!fileStats.isFile() || fileStats.size <= 44)
        throw new TTSProviderError('AUDIO_OUTPUT_INVALID', 'TTS 输出 WAV 为空或不是普通文件。');
      const wave = parsePcmWave(await readFile(audioPath));
      if (
        wave.sampleRate !== result.sampleRate
        || wave.channels !== result.channels
        || wave.frameCount !== result.frameCount
        || Math.abs(wave.durationSeconds - result.durationSeconds) > 0.001
      ) {
        throw new TTSProviderError('AUDIO_OUTPUT_INVALID', 'TTS Sidecar 元数据与 WAV 文件不一致。', {
          details: { response: result, wave },
        });
      }
      return result;
    } catch (error) {
      await rm(audioPath, { force: true }).catch(() => undefined);
      throw mapSidecarError(error);
    }
  }

  async cancel(taskId: string): Promise<TTSCancelResult> {
    if (!taskId.trim())
      throw new TTSProviderError('INVALID_REQUEST', 'taskId 不能为空。');
    const result = await this.#call(() => this.#client.request(
      'cancel',
      { taskId },
      this.#timeouts.controlMs,
    ).then(parseCancelResult));
    if (result.taskId !== taskId)
      throw protocolError('TTS Sidecar 取消响应 taskId 与请求不一致。');
    return result;
  }

  async getStatus(): Promise<TTSProviderStatus> {
    if (this.#disposed)
      return disposedStatus();
    return this.#call(() => this.#client.request('status', {}, this.#timeouts.controlMs).then(parseStatus));
  }

  async dispose(): Promise<void> {
    if (this.#disposed)
      return;
    let disposalError: unknown;
    try {
      await this.#client.dispose();
    } catch (error) {
      disposalError = mapSidecarError(error);
    } finally {
      this.#disposed = true;
      if (this.#ownsOutputDirectory)
        await rm(this.#outputDirectory, { force: true, recursive: true });
    }
    if (disposalError)
      throw disposalError;
  }

  async #call<T>(operation: () => Promise<T>): Promise<T> {
    if (this.#disposed)
      throw new TTSProviderError('PROVIDER_DISPOSED', 'TTS Provider 已退出。');
    try {
      return await operation();
    } catch (error) {
      throw mapSidecarError(error);
    }
  }
}
