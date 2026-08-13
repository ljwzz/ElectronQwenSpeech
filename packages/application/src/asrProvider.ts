export const ASR_PROVIDER_ERROR_CODES = [
  'INVALID_REQUEST',
  'UNKNOWN_METHOD',
  'PYTHON_VERSION_UNSUPPORTED',
  'MPS_UNAVAILABLE',
  'MPS_DEVICE_MISMATCH',
  'INITIALIZATION_FAILED',
  'NOT_INITIALIZED',
  'AUDIO_NOT_FOUND',
  'TRANSCRIPTION_FAILED',
  'TASK_CANCELLED',
  'REQUEST_TIMEOUT',
  'SIDECAR_PROTOCOL_ERROR',
  'SIDECAR_UNAVAILABLE',
  'SIDECAR_EXITED',
  'SHUTDOWN_TIMEOUT',
  'PROVIDER_DISPOSED',
] as const;

export type ASRProviderErrorCode = typeof ASR_PROVIDER_ERROR_CODES[number];

export type ASRProviderState = 'uninitialized'
  | 'initializing'
  | 'ready'
  | 'busy'
  | 'failed'
  | 'shutting_down'
  | 'disposed';

export interface ASRProviderStatus {
  state: ASRProviderState;
  initialized: boolean;
  currentTaskId: string | null;
  queuedTaskCount: number;
  cancelRequested: boolean;
  loadCount: number;
}

export interface ASRProviderHealth extends ASRProviderStatus {
  healthy: boolean;
}

export interface ASRInitializeResult {
  state: 'ready';
  asrModelPath: string;
  alignerModelPath: string;
  asrDevice: string;
  alignerDevice: string;
  asrDtype: string;
  alignerDtype: string;
  loadCount: number;
}

export interface ASRTranscriptionRequest {
  taskId: string;
  audioPath: string;
  language?: string;
}

export interface ASRTimestamp {
  text: string;
  start: number;
  end: number;
}

export interface ASRTranscriptionResult {
  taskId: string;
  text: string;
  language?: string;
  timestamps: ASRTimestamp[];
}

export interface ASRBatchTranscriptionItemRequest {
  itemId: string;
  audioPath: string;
  language?: string;
}

export interface ASRBatchTranscriptionRequest {
  taskId: string;
  items: ASRBatchTranscriptionItemRequest[];
}

export interface ASRBatchTranscriptionItemResult {
  itemId: string;
  text: string;
  language?: string;
  timestamps: ASRTimestamp[];
}

export interface ASRBatchTranscriptionResult {
  taskId: string;
  items: ASRBatchTranscriptionItemResult[];
}

export type ASRCancelStatus = 'cancelled' | 'cancel_requested' | 'not_found';

export interface ASRCancelResult {
  taskId: string;
  status: ASRCancelStatus;
}

export interface ASRProvider {
  initialize: () => Promise<ASRInitializeResult>;
  health: () => Promise<ASRProviderHealth>;
  transcribe: (request: ASRTranscriptionRequest) => Promise<ASRTranscriptionResult>;
  cancel: (taskId: string) => Promise<ASRCancelResult>;
  getStatus: () => Promise<ASRProviderStatus>;
  dispose: () => Promise<void>;
}

export interface ASRBatchProvider extends ASRProvider {
  transcribeBatch: (request: ASRBatchTranscriptionRequest) => Promise<ASRBatchTranscriptionResult>;
}

export interface ASRProviderErrorOptions {
  retryable?: boolean;
  details?: unknown;
  cause?: unknown;
}

export class ASRProviderError extends Error {
  readonly code: ASRProviderErrorCode;
  readonly retryable: boolean;
  readonly details: unknown;

  constructor(code: ASRProviderErrorCode, message: string, options: ASRProviderErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'ASRProviderError';
    this.code = code;
    this.retryable = options.retryable ?? false;
    this.details = options.details;
  }
}

export function isASRProviderError(error: unknown): error is ASRProviderError {
  return error instanceof ASRProviderError;
}
