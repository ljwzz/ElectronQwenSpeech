export const TTS_PROVIDER_ERROR_CODES = [
  'INVALID_REQUEST',
  'UNKNOWN_METHOD',
  'PYTHON_VERSION_UNSUPPORTED',
  'MPS_UNAVAILABLE',
  'MPS_DEVICE_MISMATCH',
  'INITIALIZATION_FAILED',
  'NOT_INITIALIZED',
  'UNSUPPORTED_LANGUAGE',
  'UNSUPPORTED_VOICE',
  'SYNTHESIS_FAILED',
  'AUDIO_WRITE_FAILED',
  'AUDIO_OUTPUT_INVALID',
  'TASK_CANCELLED',
  'REQUEST_TIMEOUT',
  'SIDECAR_PROTOCOL_ERROR',
  'SIDECAR_UNAVAILABLE',
  'SIDECAR_EXITED',
  'SHUTDOWN_TIMEOUT',
  'PROVIDER_DISPOSED',
] as const;

export type TTSProviderErrorCode = typeof TTS_PROVIDER_ERROR_CODES[number];

export type TTSProviderState = 'uninitialized'
  | 'initializing'
  | 'ready'
  | 'busy'
  | 'failed'
  | 'shutting_down'
  | 'disposed';

export interface TTSProviderStatus {
  state: TTSProviderState;
  initialized: boolean;
  currentTaskId: string | null;
  queuedTaskCount: number;
  cancelRequested: boolean;
  loadCount: number;
}

export interface TTSProviderHealth extends TTSProviderStatus {
  healthy: boolean;
}

export interface TTSInitializeResult {
  state: 'ready';
  modelPath: string;
  device: string;
  dtype: string;
  supportedLanguages: string[];
  supportedVoiceIds: string[];
  loadCount: number;
}

export interface TTSSynthesisRequest {
  taskId: string;
  text: string;
  language: string;
  voiceId: string;
  instruction?: string;
}

export interface TTSSynthesisResult {
  taskId: string;
  audioPath: string;
  mimeType: 'audio/wav';
  sampleRate: number;
  channels: number;
  frameCount: number;
  durationSeconds: number;
  generationSeconds: number;
}

export type TTSCancelStatus = 'cancelled' | 'cancel_requested' | 'not_found';

export interface TTSCancelResult {
  taskId: string;
  status: TTSCancelStatus;
}

export interface TTSProvider {
  initialize: () => Promise<TTSInitializeResult>;
  health: () => Promise<TTSProviderHealth>;
  synthesize: (request: TTSSynthesisRequest) => Promise<TTSSynthesisResult>;
  cancel: (taskId: string) => Promise<TTSCancelResult>;
  getStatus: () => Promise<TTSProviderStatus>;
  dispose: () => Promise<void>;
}

export interface TTSProviderErrorOptions {
  retryable?: boolean;
  details?: unknown;
  cause?: unknown;
}

export class TTSProviderError extends Error {
  readonly code: TTSProviderErrorCode;
  readonly retryable: boolean;
  readonly details: unknown;

  constructor(code: TTSProviderErrorCode, message: string, options: TTSProviderErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'TTSProviderError';
    this.code = code;
    this.retryable = options.retryable ?? false;
    this.details = options.details;
  }
}

export function isTTSProviderError(error: unknown): error is TTSProviderError {
  return error instanceof TTSProviderError;
}
