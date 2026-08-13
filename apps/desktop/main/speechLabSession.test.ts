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
  TTSCancelResult,
  TTSInitializeResult,
  TTSProvider,
  TTSProviderHealth,
  TTSProviderStatus,
  TTSSynthesisRequest,
  TTSSynthesisResult,
} from '@electron-qwen-speech/application';

import { access, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { ASRProviderError, TTSProviderError } from '@electron-qwen-speech/application';
import { describe, expect, it } from 'vitest';
import { SpeechLabSession } from './speechLabSession.ts';

const TTS_REQUEST = {
  text: '银行行长走过人行道。',
  voiceId: 'Vivian',
} as const;

function asrInitialization(): ASRInitializeResult {
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

function ttsInitialization(): TTSInitializeResult {
  return {
    state: 'ready',
    runtime: 'mlx',
    accelerator: 'metal',
    modelPath: '/models/tts',
    device: 'gpu:0',
    modelFormat: 'mlx',
    modelRevision: 'tts-revision',
    modelPrecision: 'bfloat16',
    speechTokenizerPrecision: 'float32',
    dtype: 'bfloat16',
    supportedLanguages: ['Chinese'],
    supportedVoiceIds: ['Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric'],
    loadCount: 1,
  };
}

function asrStatus(): ASRProviderStatus {
  return {
    state: 'ready',
    initialized: true,
    currentTaskId: null,
    queuedTaskCount: 0,
    cancelRequested: false,
    loadCount: 1,
  };
}

function ttsStatus(): TTSProviderStatus {
  return {
    state: 'ready',
    initialized: true,
    currentTaskId: null,
    queuedTaskCount: 0,
    cancelRequested: false,
    loadCount: 1,
  };
}

class FakeASRProvider implements ASRBatchProvider {
  readonly transcribeCalls: ASRTranscriptionRequest[] = [];
  initializeCalls = 0;
  disposeCalls = 0;
  cancelCalls: string[] = [];
  onInitialize: (() => Promise<ASRInitializeResult>) | undefined;
  onTranscribe: ((request: ASRTranscriptionRequest) => Promise<ASRTranscriptionResult>) | undefined;

  constructor(readonly events: string[] = []) {}

  async initialize(): Promise<ASRInitializeResult> {
    this.initializeCalls += 1;
    this.events.push('asr:initialize');
    return this.onInitialize ? this.onInitialize() : asrInitialization();
  }

  async health(): Promise<ASRProviderHealth> {
    return { ...asrStatus(), healthy: true };
  }

  async transcribe(request: ASRTranscriptionRequest): Promise<ASRTranscriptionResult> {
    this.transcribeCalls.push(request);
    this.events.push('asr:transcribe');
    return this.onTranscribe
      ? this.onTranscribe(request)
      : {
          taskId: request.taskId,
          text: TTS_REQUEST.text,
          language: 'Chinese',
          timestamps: [{ text: TTS_REQUEST.text, start: 0, end: 1 }],
          alignmentStatus: 'aligned',
        };
  }

  async transcribeBatch(request: ASRBatchTranscriptionRequest): Promise<ASRBatchTranscriptionResult> {
    return {
      taskId: request.taskId,
      items: request.items.map(item => ({
        itemId: item.itemId,
        text: TTS_REQUEST.text,
        language: 'Chinese',
        timestamps: [],
        alignmentStatus: 'unsupported_language',
      })),
    };
  }

  async cancel(taskId: string): Promise<ASRCancelResult> {
    this.cancelCalls.push(taskId);
    return { taskId, status: 'cancel_requested' };
  }

  async getStatus(): Promise<ASRProviderStatus> {
    return asrStatus();
  }

  async dispose(): Promise<void> {
    this.disposeCalls += 1;
    this.events.push('asr:dispose');
  }
}

class FakeTTSProvider implements TTSProvider {
  readonly synthesisCalls: TTSSynthesisRequest[] = [];
  readonly audioPaths: string[] = [];
  initializeCalls = 0;
  disposeCalls = 0;
  getStatusCalls = 0;
  status: TTSProviderStatus = ttsStatus();
  cancelCalls: string[] = [];
  onInitialize: (() => Promise<TTSInitializeResult>) | undefined;
  onSynthesize: ((request: TTSSynthesisRequest) => Promise<TTSSynthesisResult>) | undefined;

  constructor(
    readonly outputDirectory: string,
    readonly events: string[] = [],
  ) {}

  async initialize(): Promise<TTSInitializeResult> {
    this.initializeCalls += 1;
    this.events.push('tts:initialize');
    return this.onInitialize ? this.onInitialize() : ttsInitialization();
  }

  async health(): Promise<TTSProviderHealth> {
    return { ...ttsStatus(), healthy: true };
  }

  async synthesize(request: TTSSynthesisRequest): Promise<TTSSynthesisResult> {
    this.synthesisCalls.push(request);
    this.events.push('tts:synthesize');
    if (this.onSynthesize)
      return this.onSynthesize(request);
    const audioPath = path.join(this.outputDirectory, `${request.taskId}.wav`);
    await writeFile(audioPath, new Uint8Array([82, 73, 70, 70]));
    this.audioPaths.push(audioPath);
    return {
      taskId: request.taskId,
      audioPath,
      mimeType: 'audio/wav',
      sampleRate: 24_000,
      channels: 1,
      frameCount: 24_000,
      durationSeconds: 1,
      generationSeconds: 0.25,
    };
  }

  async cancel(taskId: string): Promise<TTSCancelResult> {
    this.cancelCalls.push(taskId);
    return { taskId, status: 'cancel_requested' };
  }

  async getStatus(): Promise<TTSProviderStatus> {
    this.getStatusCalls += 1;
    return { ...this.status };
  }

  async dispose(): Promise<void> {
    this.disposeCalls += 1;
    this.events.push('tts:dispose');
  }
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

describe('speechLabSession', () => {
  it('resident 初始化 TTS 与 ASR，并在其中一项失败时保留另一项能力', async () => {
    const tts = new FakeTTSProvider('/tmp/not-used');
    tts.onInitialize = async () => {
      throw new TTSProviderError('ACCELERATOR_UNAVAILABLE', 'TTS accelerator unavailable');
    };
    const asr = new FakeASRProvider();
    const session = new SpeechLabSession({
      asrProviderFactory: () => asr,
      ttsProviderFactory: () => tts,
    });

    const snapshot = await session.initialize();

    expect(snapshot.operation).toBe('idle');
    expect(snapshot.tts).toMatchObject({ state: 'failed', initialized: false });
    expect(snapshot.asr).toMatchObject({ state: 'ready', initialized: true, leaseMode: 'resident' });
    expect(tts.disposeCalls).toBe(1);
    await session.dispose();
  });

  it('停用会等待进行中的初始化并在完成后释放两个 Provider', async () => {
    let resolveTTSInitialization: ((value: TTSInitializeResult) => void) | undefined;
    const asr = new FakeASRProvider();
    const tts = new FakeTTSProvider('/tmp/not-used');
    tts.onInitialize = () => new Promise((resolve) => {
      resolveTTSInitialization = resolve;
    });
    const session = new SpeechLabSession({
      asrProviderFactory: () => asr,
      ttsProviderFactory: () => tts,
    });
    const initialization = session.initialize();
    await flushPromises();

    let stopSettled = false;
    const stopped = session.stop().then((value) => {
      stopSettled = true;
      return value;
    });
    await flushPromises();

    expect(stopSettled).toBe(false);
    expect(session.snapshot().operation).toBe('stopping');
    resolveTTSInitialization?.(ttsInitialization());
    await initialization;
    await stopped;

    expect(tts.disposeCalls).toBe(1);
    expect(asr.initializeCalls).toBe(1);
    expect(asr.disposeCalls).toBe(1);
    expect(session.snapshot()).toMatchObject({
      operation: 'idle',
      tts: { initialized: false, state: 'uninitialized' },
      asr: { initialized: false, state: 'uninitialized' },
    });
    await session.dispose();
  });

  it('停用会等待进行中的驻留策略切换并阻止迟到初始化重新驻留', async () => {
    let resolveTTSInitialization: ((value: TTSInitializeResult) => void) | undefined;
    const asr = new FakeASRProvider();
    const tts = new FakeTTSProvider('/tmp/not-used');
    tts.onInitialize = () => new Promise((resolve) => {
      resolveTTSInitialization = resolve;
    });
    const session = new SpeechLabSession({
      asrProviderFactory: () => asr,
      ttsProviderFactory: () => tts,
    });
    await session.setRuntimePolicy('exclusive');
    const switching = session.setRuntimePolicy('resident');
    await flushPromises();

    let stopSettled = false;
    const stopped = session.stop().then((value) => {
      stopSettled = true;
      return value;
    });
    await flushPromises();

    expect(stopSettled).toBe(false);
    expect(session.snapshot().operation).toBe('stopping');
    resolveTTSInitialization?.(ttsInitialization());
    await switching;
    await stopped;

    expect(tts.disposeCalls).toBe(1);
    expect(asr.initializeCalls).toBe(1);
    expect(asr.disposeCalls).toBe(1);
    expect(session.snapshot()).toMatchObject({
      policy: 'resident',
      operation: 'idle',
      tts: { initialized: false, state: 'uninitialized' },
      asr: { initialized: false, state: 'uninitialized' },
    });
    await session.dispose();
  });

  it('新合成结果替换旧文件，clipId 只允许读取当前音频，退出时清理会话目录', async () => {
    let outputDirectory = '';
    let tts: FakeTTSProvider | undefined;
    const session = new SpeechLabSession({
      asrProviderFactory: () => new FakeASRProvider(),
      ttsProviderFactory: (directory) => {
        outputDirectory = directory;
        tts = new FakeTTSProvider(directory);
        return tts;
      },
    });

    const first = await session.synthesize(TTS_REQUEST);
    const firstPath = tts!.audioPaths[0]!;
    const firstAudio = await session.readGeneratedAudio(first.clipId);
    expect([...firstAudio.data]).toEqual([82, 73, 70, 70]);

    const second = await session.synthesize({ ...TTS_REQUEST, text: '重庆银行。' });
    await expect(access(firstPath)).rejects.toBeDefined();
    await expect(session.readGeneratedAudio(first.clipId)).rejects.toMatchObject({ code: 'FORBIDDEN' });
    await expect(session.readGeneratedAudio(second.clipId)).resolves.toMatchObject({ clipId: second.clipId });

    await session.dispose();
    await expect(access(outputDirectory)).rejects.toBeDefined();
  });

  it('tts sidecar 新进程未初始化时不复用旧缓存并重新初始化', async () => {
    let tts: FakeTTSProvider | undefined;
    const session = new SpeechLabSession({
      asrProviderFactory: () => new FakeASRProvider(),
      ttsProviderFactory: (directory) => {
        tts = new FakeTTSProvider(directory);
        return tts;
      },
    });
    await session.synthesize(TTS_REQUEST);
    tts!.status = {
      state: 'uninitialized',
      initialized: false,
      currentTaskId: null,
      queuedTaskCount: 0,
      cancelRequested: false,
      loadCount: 0,
    };

    await session.synthesize({ ...TTS_REQUEST, text: '重庆银行。' });

    expect(tts!.getStatusCalls).toBe(1);
    expect(tts!.initializeCalls).toBe(2);
    expect(session.snapshot().tts).toMatchObject({ initialized: true, state: 'ready' });
    await session.dispose();
  });

  it('resident 闭环保持两个 Provider 驻留并给出文本一致结论', async () => {
    const asr = new FakeASRProvider();
    let tts: FakeTTSProvider | undefined;
    const session = new SpeechLabSession({
      asrProviderFactory: () => asr,
      ttsProviderFactory: (directory) => {
        tts = new FakeTTSProvider(directory);
        return tts;
      },
    });

    const report = await session.runRoundTrip(TTS_REQUEST);

    expect(report).toMatchObject({
      verdict: 'consistent',
      transcription: { normalizedMatch: true, failure: null },
    });
    expect(tts!.disposeCalls).toBe(0);
    expect(asr.disposeCalls).toBe(0);
    expect(session.snapshot()).toMatchObject({
      policy: 'resident',
      tts: { initialized: true },
      asr: { initialized: true, leaseMode: 'resident' },
    });
    await session.dispose();
  });

  it('exclusive 闭环按 ASR 释放、TTS 推理、TTS 释放、ASR 推理顺序运行', async () => {
    const events: string[] = [];
    const asrProviders: FakeASRProvider[] = [];
    const ttsProviders: FakeTTSProvider[] = [];
    const session = new SpeechLabSession({
      asrProviderFactory: () => {
        const provider = new FakeASRProvider(events);
        asrProviders.push(provider);
        return provider;
      },
      ttsProviderFactory: (directory) => {
        const provider = new FakeTTSProvider(directory, events);
        ttsProviders.push(provider);
        return provider;
      },
    });
    await session.setRuntimePolicy('exclusive');
    await session.initialize();
    events.length = 0;

    const report = await session.runRoundTrip(TTS_REQUEST);

    expect(report.verdict).toBe('consistent');
    expect(events).toEqual([
      'asr:dispose',
      'tts:initialize',
      'tts:synthesize',
      'tts:dispose',
      'asr:initialize',
      'asr:transcribe',
    ]);
    expect(ttsProviders).toHaveLength(1);
    expect(asrProviders).toHaveLength(2);
    expect(session.snapshot()).toMatchObject({
      policy: 'exclusive',
      tts: { initialized: false },
      asr: { initialized: true },
    });
    await session.dispose();
  });

  it('策略切换返回最终 idle 快照，避免 Renderer 覆盖最终事件', async () => {
    const session = new SpeechLabSession({
      asrProviderFactory: () => new FakeASRProvider(),
      ttsProviderFactory: directory => new FakeTTSProvider(directory),
    });

    const exclusive = await session.setRuntimePolicy('exclusive');
    const resident = await session.setRuntimePolicy('resident');

    expect(exclusive).toMatchObject({ policy: 'exclusive', operation: 'idle' });
    expect(resident).toMatchObject({ policy: 'resident', operation: 'idle' });
    await session.dispose();
  });

  it('任务运行中拒绝切换策略，并将取消请求发送给当前 TTS 任务', async () => {
    let resolveSynthesis: ((value: TTSSynthesisResult) => void) | undefined;
    let tts: FakeTTSProvider | undefined;
    const session = new SpeechLabSession({
      asrProviderFactory: () => new FakeASRProvider(),
      ttsProviderFactory: (directory) => {
        tts = new FakeTTSProvider(directory);
        tts.onSynthesize = _request => new Promise((resolve) => {
          resolveSynthesis = resolve;
        });
        return tts;
      },
    });
    const synthesis = session.synthesize(TTS_REQUEST);
    await flushPromises();

    await expect(session.setRuntimePolicy('exclusive')).rejects.toMatchObject({ code: 'SESSION_BUSY' });
    await expect(session.cancel()).resolves.toBe(true);
    const request = tts!.synthesisCalls[0]!;
    const audioPath = path.join(tts!.outputDirectory, `${request.taskId}.wav`);
    await writeFile(audioPath, new Uint8Array([1]));
    resolveSynthesis?.({
      taskId: request.taskId,
      audioPath,
      mimeType: 'audio/wav',
      sampleRate: 24_000,
      channels: 1,
      frameCount: 1,
      durationSeconds: 1 / 24_000,
      generationSeconds: 0.1,
    });
    await synthesis;

    expect(tts!.cancelCalls).toEqual([request.taskId]);
    await session.dispose();
  });

  it('失败后保留当前音频，重试 ASR 时复用同一路径并更新结论', async () => {
    const asr = new FakeASRProvider();
    let attempt = 0;
    asr.onTranscribe = async (request) => {
      attempt += 1;
      if (attempt === 1)
        throw new ASRProviderError('TRANSCRIPTION_FAILED', 'first attempt failed');
      return {
        taskId: request.taskId,
        text: TTS_REQUEST.text,
        language: 'Chinese',
        timestamps: [],
        alignmentStatus: 'unsupported_language',
      };
    };
    const session = new SpeechLabSession({
      asrProviderFactory: () => asr,
      ttsProviderFactory: directory => new FakeTTSProvider(directory),
    });
    const synthesis = await session.synthesize(TTS_REQUEST);

    const failed = await session.transcribeCurrentClip();
    const retried = await session.transcribeCurrentClip();

    expect(failed).toMatchObject({ verdict: 'validation_failed' });
    expect(failed.transcription.failure).toMatchObject({
      code: 'TRANSCRIPTION_FAILED',
      stage: 'asr_transcribe',
    });
    expect(retried).toMatchObject({ verdict: 'consistent' });
    expect(asr.transcribeCalls.map(call => call.audioPath)).toEqual([
      asr.transcribeCalls[0]!.audioPath,
      asr.transcribeCalls[0]!.audioPath,
    ]);
    await expect(session.readGeneratedAudio(synthesis.clipId)).resolves.toMatchObject({
      clipId: synthesis.clipId,
    });
    await session.dispose();
  });
});
