import type {
  ASRAlignmentStatus,
  ASRBatchProvider,
  ASRBatchTranscriptionItemResult,
  ASRBatchTranscriptionRequest,
  ASRBatchTranscriptionResult,
  ASRCancelResult,
  ASRInitializeResult,
  ASRProviderErrorCode,
  ASRProviderHealth,
  ASRProviderState,
  ASRProviderStatus,
  ASRTimestamp,
  ASRTranscriptionRequest,
  ASRTranscriptionResult,
  SpeechModelFormat,
  SpeechModelPrecision,
  SpeechRuntime,
} from '@electron-qwen-speech/application';
import type { ChildProcessWithoutNullStreams } from 'node:child_process';

import { Buffer } from 'node:buffer';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { StringDecoder } from 'node:string_decoder';
import {
  ASR_PROVIDER_ERROR_CODES,
  ASRProviderError,
} from '@electron-qwen-speech/application';
import { findDevelopmentRepositoryRoot } from '../developmentRepositoryRoot.ts';

const REPOSITORY_ROOT = findDevelopmentRepositoryRoot();
const ALIGNER_MODEL_PATH_ENV = 'ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH';
const ASR_MODEL_PATH_ENV = 'ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH';
const MAX_STDOUT_BUFFER_BYTES = 4 * 1024 * 1024;
const MAX_STDERR_BUFFER_BYTES = 16 * 1024;
const MODEL_PRECISION_SET = new Set<SpeechModelPrecision>(['bfloat16', 'float32']);

export const QWEN_LOCAL_ASR_DEFAULTS = {
  runtime: 'mlx' as SpeechRuntime,
  pythonExecutable: path.join(REPOSITORY_ROOT, 'services', 'asr-sidecar', '.venv-mlx', 'bin', 'python'),
  accelerator: 'metal',
  device: 'gpu:0',
  modelFormat: 'mlx' as SpeechModelFormat,
  asrModelRevision: 'e1f6c266914abc5a46e8756e02580f834a6cf8a7',
  alignerModelRevision: '53c8c0e46733eec430e4b53dd6471d0e5dee45f8',
  asrModelPrecision: 'bfloat16' as SpeechModelPrecision,
  alignerModelPrecision: 'bfloat16' as SpeechModelPrecision,
  sidecarEntryPoint: path.join(REPOSITORY_ROOT, 'services', 'asr-sidecar', 'main.py'),
  workingDirectory: REPOSITORY_ROOT,
  timeouts: {
    controlMs: 5_000,
    initializeMs: 300_000,
    transcribeMs: 180_000,
    shutdownMs: 15_000,
  },
} as const;

export interface QwenLocalASRTimeouts {
  controlMs: number;
  initializeMs: number;
  transcribeMs: number;
  shutdownMs: number;
}

export interface QwenLocalASRProviderOptions {
  pythonExecutable?: string;
  sidecarEntryPoint?: string;
  workingDirectory?: string;
  timeouts?: Partial<QwenLocalASRTimeouts>;
  environment?: NodeJS.ProcessEnv;
  spawnProcess?: typeof spawn;
  onStderr?: (output: string) => void;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: ASRProviderError) => void;
  timeout: NodeJS.Timeout;
}

type SidecarMethod = 'initialize' | 'health' | 'status' | 'transcribe' | 'cancel' | 'shutdown';

const ASR_ERROR_CODE_SET = new Set<string>(ASR_PROVIDER_ERROR_CODES);
const ASR_PROVIDER_STATE_SET = new Set<ASRProviderState>([
  'uninitialized',
  'initializing',
  'ready',
  'busy',
  'failed',
  'shutting_down',
  'disposed',
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function protocolError(message: string, details?: unknown): ASRProviderError {
  return new ASRProviderError('SIDECAR_PROTOCOL_ERROR', message, { details });
}

function readString(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== 'string')
    throw protocolError(`Sidecar 响应字段 ${key} 必须是字符串。`, record);
  return value;
}

function readNullableString(record: Record<string, unknown>, key: string): string | null {
  const value = record[key];
  if (value !== null && typeof value !== 'string')
    throw protocolError(`Sidecar 响应字段 ${key} 必须是字符串或 null。`, record);
  return value;
}

function readRuntime(record: Record<string, unknown>): SpeechRuntime {
  const runtime = readString(record, 'runtime');
  if (runtime !== 'mlx')
    throw protocolError(`Sidecar 返回未知运行时：${runtime}`, record);
  return 'mlx';
}

function readModelFormat(record: Record<string, unknown>): SpeechModelFormat {
  const modelFormat = readString(record, 'modelFormat');
  if (modelFormat !== 'mlx')
    throw protocolError(`Sidecar 返回未知模型格式：${modelFormat}`, record);
  return 'mlx';
}

function readModelPrecision(record: Record<string, unknown>, key: string): SpeechModelPrecision {
  const precision = readString(record, key) as SpeechModelPrecision;
  if (!MODEL_PRECISION_SET.has(precision))
    throw protocolError(`Sidecar 返回未知模型精度：${precision}`, record);
  return precision;
}

function readAlignmentStatus(record: Record<string, unknown>): ASRAlignmentStatus {
  const alignmentStatus = readString(record, 'alignmentStatus');
  if (alignmentStatus !== 'aligned' && alignmentStatus !== 'unsupported_language')
    throw protocolError(`Sidecar 返回未知对齐状态：${alignmentStatus}`, record);
  return alignmentStatus;
}

function readBoolean(record: Record<string, unknown>, key: string): boolean {
  const value = record[key];
  if (typeof value !== 'boolean')
    throw protocolError(`Sidecar 响应字段 ${key} 必须是布尔值。`, record);
  return value;
}

function readNonNegativeInteger(record: Record<string, unknown>, key: string): number {
  const value = record[key];
  if (!Number.isInteger(value) || (value as number) < 0)
    throw protocolError(`Sidecar 响应字段 ${key} 必须是非负整数。`, record);
  return value as number;
}

function parseStatus(value: unknown): ASRProviderStatus {
  if (!isRecord(value))
    throw protocolError('Sidecar 状态响应必须是对象。', value);

  const state = readString(value, 'state') as ASRProviderState;
  if (!ASR_PROVIDER_STATE_SET.has(state))
    throw protocolError(`Sidecar 返回未知状态：${state}`, value);

  const currentTaskId = value.currentTaskId;
  if (currentTaskId !== null && typeof currentTaskId !== 'string')
    throw protocolError('Sidecar 响应字段 currentTaskId 必须是字符串或 null。', value);

  return {
    state,
    initialized: readBoolean(value, 'initialized'),
    currentTaskId,
    queuedTaskCount: readNonNegativeInteger(value, 'queuedTaskCount'),
    cancelRequested: readBoolean(value, 'cancelRequested'),
    loadCount: readNonNegativeInteger(value, 'loadCount'),
  };
}

function parseHealth(value: unknown): ASRProviderHealth {
  if (!isRecord(value))
    throw protocolError('Sidecar 健康响应必须是对象。', value);
  return {
    ...parseStatus(value),
    healthy: readBoolean(value, 'healthy'),
  };
}

function parseInitializeResult(value: unknown): ASRInitializeResult {
  if (!isRecord(value))
    throw protocolError('Sidecar 初始化响应必须是对象。', value);
  if (value.state !== 'ready')
    throw protocolError('Sidecar 初始化完成后必须处于 ready。', value);

  return {
    state: 'ready',
    runtime: readRuntime(value),
    accelerator: readString(value, 'accelerator') as ASRInitializeResult['accelerator'],
    device: readString(value, 'device') as ASRInitializeResult['device'],
    modelFormat: readModelFormat(value),
    asrModelPath: readString(value, 'asrModelPath'),
    alignerModelPath: readString(value, 'alignerModelPath'),
    asrModelRevision: readNullableString(value, 'asrModelRevision') as ASRInitializeResult['asrModelRevision'],
    alignerModelRevision: readNullableString(value, 'alignerModelRevision') as ASRInitializeResult['alignerModelRevision'],
    asrModelPrecision: readModelPrecision(value, 'asrModelPrecision') as ASRInitializeResult['asrModelPrecision'],
    alignerModelPrecision: readModelPrecision(value, 'alignerModelPrecision') as ASRInitializeResult['alignerModelPrecision'],
    asrDevice: readString(value, 'asrDevice') as ASRInitializeResult['asrDevice'],
    alignerDevice: readString(value, 'alignerDevice') as ASRInitializeResult['alignerDevice'],
    asrDtype: readString(value, 'asrDtype') as ASRInitializeResult['asrDtype'],
    alignerDtype: readString(value, 'alignerDtype') as ASRInitializeResult['alignerDtype'],
    loadCount: readNonNegativeInteger(value, 'loadCount'),
  };
}

function assertRuntimeMetadata(
  result: ASRInitializeResult,
  asrModelPath: string,
  alignerModelPath: string,
): void {
  const expected = QWEN_LOCAL_ASR_DEFAULTS;
  if (
    result.runtime !== expected.runtime
    || result.accelerator !== expected.accelerator
    || result.device !== expected.device
    || result.modelFormat !== expected.modelFormat
    || result.asrDevice !== expected.device
    || result.alignerDevice !== expected.device
  ) {
    throw new ASRProviderError('RUNTIME_DEVICE_MISMATCH', 'ASR Sidecar 运行时或设备与请求不一致。', {
      details: { expected, actual: result },
    });
  }
  if (
    path.resolve(result.asrModelPath) !== asrModelPath
    || path.resolve(result.alignerModelPath) !== alignerModelPath
    || result.asrModelRevision !== expected.asrModelRevision
    || result.alignerModelRevision !== expected.alignerModelRevision
    || result.asrModelPrecision !== expected.asrModelPrecision
    || result.alignerModelPrecision !== expected.alignerModelPrecision
    || result.asrDtype !== result.asrModelPrecision
    || result.alignerDtype !== result.alignerModelPrecision
  ) {
    throw new ASRProviderError('MODEL_INTEGRITY_FAILED', 'ASR Sidecar 模型身份或精度与固定配置不一致。', {
      details: { expected, actual: result },
    });
  }
}

function parseTimestamp(value: unknown): ASRTimestamp {
  if (!isRecord(value))
    throw protocolError('Sidecar 时间戳必须是对象。', value);
  const text = readString(value, 'text');
  const start = value.start;
  const end = value.end;
  if (typeof start !== 'number' || !Number.isFinite(start) || start < 0)
    throw protocolError('Sidecar 时间戳 start 必须是非负有限秒数。', value);
  if (typeof end !== 'number' || !Number.isFinite(end) || end < start)
    throw protocolError('Sidecar 时间戳 end 必须是不小于 start 的有限秒数。', value);
  return { text, start, end };
}

function parseTranscriptionResult(value: unknown): ASRTranscriptionResult {
  if (!isRecord(value))
    throw protocolError('Sidecar 转写响应必须是对象。', value);
  if (!Array.isArray(value.timestamps))
    throw protocolError('Sidecar 转写响应 timestamps 必须是数组。', value);

  const timestamps = value.timestamps.map(parseTimestamp);
  for (let index = 1; index < timestamps.length; index += 1) {
    const previous = timestamps[index - 1]!;
    const current = timestamps[index]!;
    if (current.start < previous.start || current.end < previous.end)
      throw protocolError('Sidecar 时间戳必须按 start 和 end 单调递增。', value);
  }

  const alignmentStatus = readAlignmentStatus(value);
  if (alignmentStatus === 'unsupported_language' && timestamps.length > 0)
    throw protocolError('Sidecar 不支持对齐语言时 timestamps 必须为空。', value);
  if (alignmentStatus === 'aligned' && timestamps.length === 0)
    throw protocolError('Sidecar 对齐成功时 timestamps 不能为空。', value);

  const result: ASRTranscriptionResult = {
    taskId: readString(value, 'taskId'),
    text: readString(value, 'text'),
    timestamps,
    alignmentStatus,
  };
  if (value.language !== undefined) {
    if (typeof value.language !== 'string')
      throw protocolError('Sidecar 转写响应 language 必须是字符串。', value);
    result.language = value.language;
  }
  return result;
}

function parseBatchTranscriptionItemResult(value: unknown): ASRBatchTranscriptionItemResult {
  if (!isRecord(value))
    throw protocolError('Sidecar 批量转写项目必须是对象。', value);
  const parsed = parseTranscriptionResult({
    taskId: 'batch-item',
    text: value.text,
    language: value.language,
    timestamps: value.timestamps,
    alignmentStatus: value.alignmentStatus,
  });
  const result: ASRBatchTranscriptionItemResult = {
    itemId: readString(value, 'itemId'),
    text: parsed.text,
    timestamps: parsed.timestamps,
    alignmentStatus: parsed.alignmentStatus,
  };
  if (parsed.language !== undefined)
    result.language = parsed.language;
  return result;
}

function parseBatchTranscriptionResult(value: unknown): ASRBatchTranscriptionResult {
  if (!isRecord(value))
    throw protocolError('Sidecar 批量转写响应必须是对象。', value);
  if (!Array.isArray(value.items) || value.items.length === 0)
    throw protocolError('Sidecar 批量转写响应 items 必须是非空数组。', value);
  return {
    taskId: readString(value, 'taskId'),
    items: value.items.map(parseBatchTranscriptionItemResult),
  };
}

function parseCancelResult(value: unknown): ASRCancelResult {
  if (!isRecord(value))
    throw protocolError('Sidecar 取消响应必须是对象。', value);
  const status = readString(value, 'status');
  if (status !== 'cancelled' && status !== 'cancel_requested' && status !== 'not_found')
    throw protocolError(`Sidecar 返回未知取消状态：${status}`, value);
  return {
    taskId: readString(value, 'taskId'),
    status,
  };
}

function normalizeTimeout(value: number | undefined, fallback: number, name: string): number {
  const normalized = value ?? fallback;
  if (!Number.isInteger(normalized) || normalized <= 0)
    throw new ASRProviderError('INVALID_REQUEST', `${name} 必须是正整数毫秒值。`);
  return normalized;
}

function disposedStatus(): ASRProviderStatus {
  return {
    state: 'disposed',
    initialized: false,
    currentTaskId: null,
    queuedTaskCount: 0,
    cancelRequested: false,
    loadCount: 0,
  };
}

export class QwenLocalASRProvider implements ASRBatchProvider {
  readonly #pythonExecutable: string;
  readonly #sidecarEntryPoint: string;
  readonly #workingDirectory: string;
  readonly #timeouts: QwenLocalASRTimeouts;
  readonly #environment: NodeJS.ProcessEnv;
  readonly #spawnProcess: typeof spawn;
  readonly #onStderr: ((output: string) => void) | undefined;
  readonly #asrModelPath: string;
  readonly #alignerModelPath: string;
  readonly #pending = new Map<string, PendingRequest>();

  #child: ChildProcessWithoutNullStreams | undefined;
  #stdoutBuffer = '';
  #stdoutDecoder = new StringDecoder('utf8');
  #stderrDecoder = new StringDecoder('utf8');
  #stderrTail = '';
  #requestSequence = 0;
  #disposing = false;
  #disposed = false;
  #lastStatus: ASRProviderStatus = {
    state: 'uninitialized',
    initialized: false,
    currentTaskId: null,
    queuedTaskCount: 0,
    cancelRequested: false,
    loadCount: 0,
  };

  constructor(options: QwenLocalASRProviderOptions = {}) {
    const environment = {
      ...process.env,
      ...options.environment,
    };
    const configuredASRModelPath = environment[ASR_MODEL_PATH_ENV]?.trim();
    const configuredAlignerModelPath = environment[ALIGNER_MODEL_PATH_ENV]?.trim();
    if (!configuredASRModelPath || !path.isAbsolute(configuredASRModelPath)) {
      throw new ASRProviderError(
        'MODEL_INTEGRITY_FAILED',
        `${ASR_MODEL_PATH_ENV} 必须配置为绝对模型目录。`,
        { details: { environmentVariable: ASR_MODEL_PATH_ENV } },
      );
    }
    if (!configuredAlignerModelPath || !path.isAbsolute(configuredAlignerModelPath)) {
      throw new ASRProviderError(
        'MODEL_INTEGRITY_FAILED',
        `${ALIGNER_MODEL_PATH_ENV} 必须配置为绝对模型目录。`,
        { details: { environmentVariable: ALIGNER_MODEL_PATH_ENV } },
      );
    }
    this.#asrModelPath = path.resolve(configuredASRModelPath);
    this.#alignerModelPath = path.resolve(configuredAlignerModelPath);
    this.#pythonExecutable = options.pythonExecutable ?? QWEN_LOCAL_ASR_DEFAULTS.pythonExecutable;
    this.#sidecarEntryPoint = options.sidecarEntryPoint ?? QWEN_LOCAL_ASR_DEFAULTS.sidecarEntryPoint;
    this.#workingDirectory = options.workingDirectory ?? QWEN_LOCAL_ASR_DEFAULTS.workingDirectory;
    this.#timeouts = {
      controlMs: normalizeTimeout(options.timeouts?.controlMs, QWEN_LOCAL_ASR_DEFAULTS.timeouts.controlMs, 'controlMs'),
      initializeMs: normalizeTimeout(options.timeouts?.initializeMs, QWEN_LOCAL_ASR_DEFAULTS.timeouts.initializeMs, 'initializeMs'),
      transcribeMs: normalizeTimeout(options.timeouts?.transcribeMs, QWEN_LOCAL_ASR_DEFAULTS.timeouts.transcribeMs, 'transcribeMs'),
      shutdownMs: normalizeTimeout(options.timeouts?.shutdownMs, QWEN_LOCAL_ASR_DEFAULTS.timeouts.shutdownMs, 'shutdownMs'),
    };
    this.#environment = {
      ...environment,
      HF_DATASETS_OFFLINE: '1',
      HF_HUB_DISABLE_TELEMETRY: '1',
      HF_HUB_OFFLINE: '1',
      PYTHONUNBUFFERED: '1',
      TRANSFORMERS_OFFLINE: '1',
      [ALIGNER_MODEL_PATH_ENV]: this.#alignerModelPath,
      [ASR_MODEL_PATH_ENV]: this.#asrModelPath,
    };
    this.#spawnProcess = options.spawnProcess ?? spawn;
    this.#onStderr = options.onStderr;
  }

  async initialize(): Promise<ASRInitializeResult> {
    const result = parseInitializeResult(await this.#request('initialize', {}, this.#timeouts.initializeMs));
    assertRuntimeMetadata(result, this.#asrModelPath, this.#alignerModelPath);
    this.#lastStatus = {
      state: result.state,
      initialized: true,
      currentTaskId: null,
      queuedTaskCount: 0,
      cancelRequested: false,
      loadCount: result.loadCount,
    };
    return result;
  }

  async health(): Promise<ASRProviderHealth> {
    const health = parseHealth(await this.#request('health', {}, this.#timeouts.controlMs));
    this.#lastStatus = health;
    return health;
  }

  async transcribe(request: ASRTranscriptionRequest): Promise<ASRTranscriptionResult> {
    if (!request.taskId.trim())
      throw new ASRProviderError('INVALID_REQUEST', 'taskId 不能为空。');
    if (!request.audioPath.trim())
      throw new ASRProviderError('INVALID_REQUEST', 'audioPath 不能为空。');
    if (request.language !== undefined && !request.language.trim())
      throw new ASRProviderError('INVALID_REQUEST', 'language 不能为空字符串。');

    const params: Record<string, unknown> = {
      taskId: request.taskId,
      audioPath: request.audioPath,
    };
    if (request.language !== undefined)
      params.language = request.language;

    const result = parseTranscriptionResult(await this.#request('transcribe', params, this.#timeouts.transcribeMs));
    if (result.taskId !== request.taskId)
      throw protocolError('Sidecar 转写响应 taskId 与请求不一致。', { expected: request.taskId, actual: result.taskId });
    return result;
  }

  async transcribeBatch(request: ASRBatchTranscriptionRequest): Promise<ASRBatchTranscriptionResult> {
    if (!request.taskId.trim())
      throw new ASRProviderError('INVALID_REQUEST', 'taskId 不能为空。');
    if (!Array.isArray(request.items) || request.items.length === 0)
      throw new ASRProviderError('INVALID_REQUEST', 'items 必须是非空数组。');

    const seenItemIds = new Set<string>();
    for (const item of request.items) {
      if (!item.itemId.trim())
        throw new ASRProviderError('INVALID_REQUEST', 'itemId 不能为空。');
      if (seenItemIds.has(item.itemId))
        throw new ASRProviderError('INVALID_REQUEST', `itemId 重复：${item.itemId}`);
      seenItemIds.add(item.itemId);
      if (!item.audioPath.trim())
        throw new ASRProviderError('INVALID_REQUEST', 'audioPath 不能为空。');
      if (item.language !== undefined && !item.language.trim())
        throw new ASRProviderError('INVALID_REQUEST', 'language 不能为空字符串。');
    }

    const result = parseBatchTranscriptionResult(await this.#request('transcribe', {
      taskId: request.taskId,
      items: request.items.map(item => ({
        itemId: item.itemId,
        audioPath: item.audioPath,
        ...(item.language === undefined ? {} : { language: item.language }),
      })),
    }, this.#timeouts.transcribeMs));
    if (result.taskId !== request.taskId)
      throw protocolError('Sidecar 批量转写响应 taskId 与请求不一致。', { expected: request.taskId, actual: result.taskId });
    const expectedItemIds = request.items.map(item => item.itemId);
    const actualItemIds = result.items.map(item => item.itemId);
    if (actualItemIds.some((itemId, index) => itemId !== expectedItemIds[index])) {
      throw protocolError('Sidecar 批量转写响应项目顺序或 itemId 与请求不一致。', {
        expected: expectedItemIds,
        actual: actualItemIds,
      });
    }
    return result;
  }

  async cancel(taskId: string): Promise<ASRCancelResult> {
    if (!taskId.trim())
      throw new ASRProviderError('INVALID_REQUEST', 'taskId 不能为空。');
    const result = parseCancelResult(await this.#request('cancel', { taskId }, this.#timeouts.controlMs));
    if (result.taskId !== taskId)
      throw protocolError('Sidecar 取消响应 taskId 与请求不一致。', { expected: taskId, actual: result.taskId });
    return result;
  }

  async getStatus(): Promise<ASRProviderStatus> {
    if (this.#disposed)
      return disposedStatus();
    const status = parseStatus(await this.#request('status', {}, this.#timeouts.controlMs));
    this.#lastStatus = status;
    return status;
  }

  async dispose(): Promise<void> {
    if (this.#disposed)
      return;
    if (this.#disposing)
      throw new ASRProviderError('PROVIDER_DISPOSED', 'ASR Provider 正在退出。');

    this.#disposing = true;
    const child = this.#child;
    if (!child) {
      this.#finishDisposal();
      return;
    }

    const deadline = Date.now() + this.#timeouts.shutdownMs;
    try {
      await this.#request('shutdown', {}, this.#timeouts.shutdownMs, true);
      const remainingMs = deadline - Date.now();
      if (remainingMs <= 0)
        throw new ASRProviderError('SHUTDOWN_TIMEOUT', 'ASR Sidecar 未在退出超时内结束。');
      await this.#waitForExit(child, remainingMs);
      this.#finishDisposal();
    } catch (error) {
      if (child.exitCode === null)
        child.kill('SIGTERM');
      this.#finishDisposal();
      if (error instanceof ASRProviderError && error.code !== 'REQUEST_TIMEOUT')
        throw error;
      throw new ASRProviderError('SHUTDOWN_TIMEOUT', 'ASR Sidecar 未在退出超时内结束，已终止该 Provider 创建的子进程。', {
        cause: error,
      });
    }
  }

  #finishDisposal(): void {
    this.#failAll(new ASRProviderError('PROVIDER_DISPOSED', 'ASR Provider 已退出。'));
    this.#disposing = false;
    this.#disposed = true;
    this.#child = undefined;
    this.#lastStatus = disposedStatus();
  }

  #assertAvailable(allowDisposing: boolean): void {
    if (this.#disposed || (this.#disposing && !allowDisposing))
      throw new ASRProviderError('PROVIDER_DISPOSED', 'ASR Provider 已退出或正在退出。');
  }

  #request(
    method: SidecarMethod,
    params: Record<string, unknown>,
    timeoutMs: number,
    allowDisposing = false,
  ): Promise<unknown> {
    this.#assertAvailable(allowDisposing);
    const child = this.#ensureProcess();
    const id = `${method}-${++this.#requestSequence}`;

    return new Promise<unknown>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.#pending.delete(id);
        reject(new ASRProviderError('REQUEST_TIMEOUT', `ASR Sidecar ${method} 请求超时。`, {
          retryable: method !== 'initialize',
          details: { method, timeoutMs },
        }));
      }, timeoutMs);

      this.#pending.set(id, { resolve, reject, timeout });
      const payload = `${JSON.stringify({ id, method, params })}\n`;
      try {
        child.stdin.write(payload, 'utf8', (error) => {
          if (error)
            this.#rejectPending(id, new ASRProviderError('SIDECAR_UNAVAILABLE', '无法写入 ASR Sidecar。', { cause: error }));
        });
      } catch (error) {
        this.#rejectPending(id, new ASRProviderError('SIDECAR_UNAVAILABLE', '无法写入 ASR Sidecar。', { cause: error }));
      }
    });
  }

  #ensureProcess(): ChildProcessWithoutNullStreams {
    if (this.#child && this.#child.exitCode === null)
      return this.#child;

    let child: ReturnType<typeof spawn>;
    try {
      child = this.#spawnProcess(
        this.#pythonExecutable,
        ['-u', this.#sidecarEntryPoint],
        {
          cwd: this.#workingDirectory,
          env: this.#environment,
          stdio: ['pipe', 'pipe', 'pipe'],
        },
      );
    } catch (error) {
      throw new ASRProviderError('SIDECAR_UNAVAILABLE', '无法启动 ASR Sidecar。', { cause: error });
    }

    if (!child.stdin || !child.stdout || !child.stderr) {
      child.kill('SIGTERM');
      throw new ASRProviderError('SIDECAR_UNAVAILABLE', 'ASR Sidecar 未提供可用的标准输入输出管道。');
    }

    const pipedChild = child as ChildProcessWithoutNullStreams;
    this.#child = pipedChild;
    this.#stdoutBuffer = '';
    this.#stdoutDecoder = new StringDecoder('utf8');
    this.#stderrDecoder = new StringDecoder('utf8');
    this.#stderrTail = '';

    pipedChild.stdout.on('data', chunk => this.#handleStdout(chunk as Buffer));
    pipedChild.stderr.on('data', chunk => this.#handleStderr(chunk as Buffer));
    pipedChild.stdin.on('error', error => this.#handleProcessFailure(
      pipedChild,
      new ASRProviderError('SIDECAR_UNAVAILABLE', 'ASR Sidecar 标准输入不可用。', { cause: error }),
    ));
    pipedChild.on('error', error => this.#handleProcessFailure(
      pipedChild,
      new ASRProviderError('SIDECAR_UNAVAILABLE', 'ASR Sidecar 启动或运行失败。', { cause: error }),
    ));
    pipedChild.on('close', (code, signal) => this.#handleClose(pipedChild, code, signal));

    return pipedChild;
  }

  #handleStdout(chunk: Buffer): void {
    this.#stdoutBuffer += this.#stdoutDecoder.write(chunk);

    let newlineIndex = this.#stdoutBuffer.indexOf('\n');
    while (newlineIndex >= 0) {
      const line = this.#stdoutBuffer.slice(0, newlineIndex).replace(/\r$/u, '');
      this.#stdoutBuffer = this.#stdoutBuffer.slice(newlineIndex + 1);
      if (Buffer.byteLength(line, 'utf8') > MAX_STDOUT_BUFFER_BYTES) {
        this.#failProtocol('ASR Sidecar 输出行超过协议上限。');
        return;
      }
      if (!line) {
        this.#failProtocol('ASR Sidecar 输出了空协议行。');
        return;
      }
      try {
        this.#handleResponse(JSON.parse(line) as unknown);
      } catch (error) {
        if (error instanceof ASRProviderError)
          this.#failProtocol(error.message, error.details);
        else
          this.#failProtocol('ASR Sidecar 输出了非法 JSON。', { line, error });
        return;
      }
      newlineIndex = this.#stdoutBuffer.indexOf('\n');
    }

    if (Buffer.byteLength(this.#stdoutBuffer, 'utf8') > MAX_STDOUT_BUFFER_BYTES)
      this.#failProtocol('ASR Sidecar 输出行超过协议上限。');
  }

  #handleResponse(value: unknown): void {
    if (!isRecord(value))
      throw protocolError('ASR Sidecar 响应必须是对象。', value);
    if (value.id !== null && typeof value.id !== 'string')
      throw protocolError('ASR Sidecar 响应 id 必须是字符串或 null。', value);
    if (value.id === null)
      throw protocolError('ASR Sidecar 返回了无法关联请求的协议错误。', value);

    const hasResult = Object.hasOwn(value, 'result');
    const hasError = Object.hasOwn(value, 'error');
    if (hasResult === hasError)
      throw protocolError('ASR Sidecar 响应必须且只能包含 result 或 error。', value);

    const pending = this.#pending.get(value.id);
    if (!pending)
      return;
    const remoteError = hasError ? this.#parseRemoteError(value.error) : undefined;
    this.#pending.delete(value.id);
    clearTimeout(pending.timeout);

    if (remoteError) {
      pending.reject(remoteError);
      return;
    }
    pending.resolve(value.result);
  }

  #parseRemoteError(value: unknown): ASRProviderError {
    if (!isRecord(value))
      throw protocolError('ASR Sidecar error 必须是对象。', value);
    const code = readString(value, 'code');
    if (!ASR_ERROR_CODE_SET.has(code))
      throw protocolError(`ASR Sidecar 返回未知错误码：${code}`, value);
    const message = readString(value, 'message');
    const retryable = value.retryable ?? false;
    if (typeof retryable !== 'boolean')
      throw protocolError('ASR Sidecar error.retryable 必须是布尔值。', value);
    return new ASRProviderError(code as ASRProviderErrorCode, message, {
      retryable,
      details: value.details,
    });
  }

  #handleStderr(chunk: Buffer): void {
    const output = this.#stderrDecoder.write(chunk);
    this.#stderrTail = `${this.#stderrTail}${output}`.slice(-MAX_STDERR_BUFFER_BYTES);
    this.#onStderr?.(output);
  }

  #failProtocol(message: string, details?: unknown): void {
    const error = protocolError(message, details);
    const child = this.#child;
    this.#failAll(error);
    if (child?.exitCode === null)
      child.kill('SIGTERM');
  }

  #handleProcessFailure(child: ChildProcessWithoutNullStreams, error: ASRProviderError): void {
    if (this.#child !== child)
      return;
    this.#failAll(error);
    if (child.exitCode === null)
      child.kill('SIGTERM');
  }

  #handleClose(
    child: ChildProcessWithoutNullStreams,
    code: number | null,
    signal: NodeJS.Signals | null,
  ): void {
    if (this.#child !== child)
      return;
    this.#child = undefined;
    if (this.#disposing)
      return;

    this.#lastStatus = {
      ...this.#lastStatus,
      state: 'failed',
      initialized: false,
      currentTaskId: null,
      queuedTaskCount: 0,
      cancelRequested: false,
    };
    this.#failAll(new ASRProviderError('SIDECAR_EXITED', 'ASR Sidecar 意外退出。', {
      retryable: true,
      details: {
        code,
        signal,
        stderr: this.#stderrTail || undefined,
      },
    }));
  }

  #rejectPending(id: string, error: ASRProviderError): void {
    const pending = this.#pending.get(id);
    if (!pending)
      return;
    this.#pending.delete(id);
    clearTimeout(pending.timeout);
    pending.reject(error);
  }

  #failAll(error: ASRProviderError): void {
    for (const [id, pending] of this.#pending) {
      this.#pending.delete(id);
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
  }

  #waitForExit(child: ChildProcessWithoutNullStreams, timeoutMs: number): Promise<void> {
    if (this.#child !== child)
      return Promise.resolve();

    return new Promise<void>((resolve, reject) => {
      let timeout: NodeJS.Timeout;
      const onExit = (): void => {
        clearTimeout(timeout);
        resolve();
      };
      timeout = setTimeout(() => {
        child.removeListener('close', onExit);
        reject(new ASRProviderError('SHUTDOWN_TIMEOUT', 'ASR Sidecar 未在退出超时内结束。'));
      }, timeoutMs);
      child.once('close', onExit);
    });
  }
}
