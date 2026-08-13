import type {
  ASRInitializeResult,
  ASRProviderErrorCode,
  ASRProviderState,
  ASRTimestamp,
  TTSInitializeResult,
  TTSProviderErrorCode,
  TTSProviderState,
} from '@electron-qwen-speech/application';

export const SPEECH_DEV_CHANNELS = {
  cancel: 'speech-dev:cancel',
  event: 'speech-dev:event',
  initialize: 'speech-dev:initialize',
  listFixtures: 'speech-dev:list-fixtures',
  readAudio: 'speech-dev:read-audio',
  readGeneratedAudio: 'speech-dev:read-generated-audio',
  refreshFixtures: 'speech-dev:refresh-fixtures',
  roundTrip: 'speech-dev:round-trip',
  run: 'speech-dev:run',
  setRuntimePolicy: 'speech-dev:set-runtime-policy',
  snapshot: 'speech-dev:snapshot',
  stop: 'speech-dev:stop',
  synthesize: 'speech-dev:synthesize',
  transcribeCurrentClip: 'speech-dev:transcribe-current-clip',
} as const;

export const SPEECH_DEV_TTS_CONFIG = Object.freeze({
  language: 'Chinese',
  defaultVoiceId: 'Vivian',
  voiceIds: Object.freeze(['Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric']),
} as const);

export const SPEECH_DEV_RUNTIME_CONFIG = Object.freeze({
  defaultPolicy: 'resident',
  policies: Object.freeze(['resident', 'exclusive']),
} as const);

export type SpeechRuntimePolicy = typeof SPEECH_DEV_RUNTIME_CONFIG.policies[number];
export type SpeechLeaseMode = 'automatic' | 'manual' | 'resident' | null;
export type SpeechSessionOperation = 'idle' | 'initializing' | 'transcribing' | 'stopping';
export type SpeechLabOperation = 'idle'
  | 'initializing'
  | 'switching'
  | 'synthesizing'
  | 'transcribing'
  | 'stopping';
export type SpeechFailureStage = 'asr_initialize'
  | 'asr_transcribe'
  | 'runtime_policy'
  | 'tts_initialize'
  | 'tts_synthesize';

export interface SpeechDevFailure {
  code: ASRProviderErrorCode
    | TTSProviderErrorCode
    | 'FIXTURE_INVALID'
    | 'FORBIDDEN'
    | 'SESSION_BUSY'
    | 'UNKNOWN';
  message: string;
  retryable: boolean;
  stage?: SpeechFailureStage;
}

export interface SpeechProviderSnapshot {
  state: ASRProviderState;
  operation: SpeechSessionOperation;
  initialized: boolean;
  activeTaskId: string | null;
  leaseMode: SpeechLeaseMode;
  idleDeadlineMs: number | null;
  initializationSeconds: number | null;
  initialization: ASRInitializeResult | null;
  lastError: SpeechDevFailure | null;
}

export interface SpeechTTSProviderSnapshot {
  state: TTSProviderState;
  initialized: boolean;
  activeTaskId: string | null;
  initializationSeconds: number | null;
  initialization: TTSInitializeResult | null;
  lastError: SpeechDevFailure | null;
}

export interface SpeechClipSummary {
  clipId: string;
  sourceText: string;
  language: string;
  voiceId: string;
  instruction: string | null;
  sampleRate: number;
  channels: number;
  frameCount: number;
  durationSeconds: number;
  generationSeconds: number;
}

export interface SpeechLabSnapshot {
  policy: SpeechRuntimePolicy;
  operation: SpeechLabOperation;
  activeTaskId: string | null;
  tts: SpeechTTSProviderSnapshot;
  asr: SpeechProviderSnapshot;
  currentClip: SpeechClipSummary | null;
  lastError: SpeechDevFailure | null;
}

export interface SpeechFixtureSummary {
  id: string;
  audioFileName: string;
  format: 'mp3' | 'wav';
  title: string | null;
  referenceText: string | null;
  error: string | null;
}

export interface SpeechFixtureAudio {
  id: string;
  mimeType: 'audio/mpeg' | 'audio/wav';
  data: Uint8Array;
}

export interface SpeechGeneratedAudio {
  clipId: string;
  mimeType: 'audio/wav';
  data: Uint8Array;
}

export interface SpeechRunRequest {
  fixtureIds: string[];
  batchSize: number;
}

export type SpeechRunItemStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
export type SpeechCpmKind = 'exact' | 'batch_derived';

export interface SpeechRunItem {
  fixtureId: string;
  audioFileName: string;
  title: string;
  referenceText: string;
  status: SpeechRunItemStatus;
  transcript: string | null;
  language: string | null;
  timestamps: ASRTimestamp[];
  similarity: number | null;
  characterCount: number;
  inferenceSeconds: number | null;
  cpm: number | null;
  cpmKind: SpeechCpmKind;
  error: SpeechDevFailure | null;
}

export interface SpeechRunReport {
  runId: string;
  requestedBatchSize: number;
  effectiveBatchSize: number;
  items: SpeechRunItem[];
  wallSeconds: number;
  inferenceSeconds: number;
  successfulCharacterCount: number;
  overallCpm: number | null;
  partial: boolean;
}

export interface SpeechSynthesisRequest {
  text: string;
  voiceId: string;
  instruction?: string;
}

export interface SpeechSynthesisReport extends SpeechClipSummary {
  wallSeconds: number;
}

export type SpeechRoundTripVerdict = 'consistent' | 'review_required' | 'validation_failed';

export interface SpeechTranscriptionReport {
  transcript: string | null;
  language: string | null;
  timestamps: ASRTimestamp[];
  normalizedMatch: boolean | null;
  similarity: number | null;
  inferenceSeconds: number | null;
  failure: SpeechDevFailure | null;
}

export interface SpeechRoundTripReport {
  runId: string;
  synthesis: SpeechSynthesisReport;
  transcription: SpeechTranscriptionReport;
  verdict: SpeechRoundTripVerdict;
  wallSeconds: number;
}

export interface SpeechRunProgressEvent {
  type: 'run-progress';
  runId: string;
  completed: number;
  total: number;
  items: SpeechRunItem[];
}

export interface SpeechSnapshotEvent {
  type: 'snapshot';
  snapshot: SpeechLabSnapshot;
}

export type SpeechDevEvent = SpeechRunProgressEvent | SpeechSnapshotEvent;

export interface SpeechDevApi {
  getSnapshot: () => Promise<SpeechLabSnapshot>;
  listFixtures: () => Promise<SpeechFixtureSummary[]>;
  refreshFixtures: () => Promise<SpeechFixtureSummary[]>;
  readFixtureAudio: (fixtureId: string) => Promise<SpeechFixtureAudio>;
  readGeneratedAudio: (clipId: string) => Promise<SpeechGeneratedAudio>;
  initialize: () => Promise<SpeechLabSnapshot>;
  setRuntimePolicy: (policy: SpeechRuntimePolicy) => Promise<SpeechLabSnapshot>;
  stop: () => Promise<SpeechLabSnapshot>;
  run: (request: SpeechRunRequest) => Promise<SpeechRunReport>;
  synthesize: (request: SpeechSynthesisRequest) => Promise<SpeechSynthesisReport>;
  transcribeCurrentClip: () => Promise<SpeechRoundTripReport>;
  runRoundTrip: (request: SpeechSynthesisRequest) => Promise<SpeechRoundTripReport>;
  cancel: () => Promise<boolean>;
  onEvent: (listener: (event: SpeechDevEvent) => void) => () => void;
}

declare global {
  interface Window {
    speechLab: SpeechDevApi;
  }
}
