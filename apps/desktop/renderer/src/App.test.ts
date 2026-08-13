import type { VueWrapper } from '@vue/test-utils';

import type {
  SpeechDevApi,
  SpeechDevEvent,
  SpeechFixtureSummary,
  SpeechLabSnapshot,
  SpeechProviderSnapshot,
  SpeechRoundTripReport,
  SpeechRunItem,
  SpeechRunReport,
  SpeechSynthesisReport,
} from '../../contracts.ts';
import { flushPromises, mount } from '@vue/test-utils';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App.vue';

const fixture: SpeechFixtureSummary = {
  id: 'fixture.wav',
  audioFileName: 'fixture.wav',
  format: 'wav',
  title: '本地识别测试',
  referenceText: '你好测试',
  error: null,
};

function providerSnapshot(overrides: Partial<SpeechProviderSnapshot> = {}): SpeechProviderSnapshot {
  return {
    state: 'ready',
    operation: 'idle',
    initialized: true,
    activeTaskId: null,
    leaseMode: 'resident',
    idleDeadlineMs: null,
    initializationSeconds: 1.25,
    initialization: {
      state: 'ready',
      asrModelPath: '/models/asr',
      alignerModelPath: '/models/aligner',
      asrDevice: 'mps:0',
      alignerDevice: 'mps:0',
      asrDtype: 'bfloat16',
      alignerDtype: 'bfloat16',
      loadCount: 1,
    },
    lastError: null,
    ...overrides,
  };
}

function labSnapshot(overrides: Partial<SpeechLabSnapshot> = {}): SpeechLabSnapshot {
  return {
    policy: 'resident',
    operation: 'idle',
    activeTaskId: null,
    tts: {
      state: 'ready',
      initialized: true,
      activeTaskId: null,
      initializationSeconds: 2.5,
      initialization: {
        state: 'ready',
        modelPath: '/models/tts',
        device: 'mps:0',
        dtype: 'bfloat16',
        supportedLanguages: ['Chinese'],
        supportedVoiceIds: ['Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric'],
        loadCount: 1,
      },
      lastError: null,
    },
    asr: providerSnapshot(),
    currentClip: null,
    lastError: null,
    ...overrides,
  };
}

function succeededItem(overrides: Partial<SpeechRunItem> = {}): SpeechRunItem {
  return {
    fixtureId: fixture.id,
    audioFileName: fixture.audioFileName,
    title: fixture.title!,
    referenceText: fixture.referenceText!,
    status: 'succeeded',
    transcript: '你好测试',
    language: 'Chinese',
    timestamps: [{ text: '你好', start: 1.25, end: 1.8 }],
    similarity: 1,
    characterCount: 4,
    inferenceSeconds: 2,
    cpm: 120,
    cpmKind: 'exact',
    error: null,
    ...overrides,
  };
}

function runReport(overrides: Partial<SpeechRunReport> = {}): SpeechRunReport {
  return {
    runId: 'run-1',
    requestedBatchSize: 1,
    effectiveBatchSize: 1,
    items: [succeededItem()],
    wallSeconds: 2.2,
    inferenceSeconds: 2,
    successfulCharacterCount: 4,
    overallCpm: 120,
    partial: false,
    ...overrides,
  };
}

function synthesisReport(overrides: Partial<SpeechSynthesisReport> = {}): SpeechSynthesisReport {
  return {
    clipId: 'clip-1',
    sourceText: '银行行长走过人行道。',
    language: 'Chinese',
    voiceId: 'Vivian',
    instruction: null,
    sampleRate: 24_000,
    channels: 1,
    frameCount: 24_000,
    durationSeconds: 1,
    generationSeconds: 0.5,
    wallSeconds: 0.8,
    ...overrides,
  };
}

function roundTripReport(overrides: Partial<SpeechRoundTripReport> = {}): SpeechRoundTripReport {
  return {
    runId: 'round-trip-1',
    synthesis: synthesisReport(),
    transcription: {
      transcript: '银行行长走过人行道。',
      language: 'Chinese',
      timestamps: [{ text: '银行行长', start: 0.25, end: 0.75 }],
      normalizedMatch: true,
      similarity: 1,
      inferenceSeconds: 1.2,
      failure: null,
    },
    verdict: 'consistent',
    wallSeconds: 2.1,
    ...overrides,
  };
}

let wrappers: VueWrapper[] = [];
let eventListener: ((event: SpeechDevEvent) => void) | undefined;
let api: SpeechDevApi;

beforeEach(() => {
  eventListener = undefined;
  let blobSequence = 0;
  vi.stubGlobal('URL', {
    ...URL,
    createObjectURL: vi.fn(() => `blob:speech-${++blobSequence}`),
    revokeObjectURL: vi.fn(),
  });
  api = {
    getSnapshot: vi.fn(async () => labSnapshot()),
    listFixtures: vi.fn(async () => [fixture]),
    refreshFixtures: vi.fn(async () => [fixture]),
    readFixtureAudio: vi.fn(async (fixtureId: string) => ({
      id: fixtureId,
      mimeType: 'audio/wav' as const,
      data: new Uint8Array([1, 2, 3]),
    })),
    readGeneratedAudio: vi.fn(async (clipId: string) => ({
      clipId,
      mimeType: 'audio/wav' as const,
      data: new Uint8Array([4, 5, 6]),
    })),
    initialize: vi.fn(async () => labSnapshot()),
    setRuntimePolicy: vi.fn(async policy => labSnapshot({ policy })),
    stop: vi.fn(async () => labSnapshot({
      tts: {
        ...labSnapshot().tts,
        state: 'uninitialized',
        initialized: false,
        initialization: null,
      },
      asr: providerSnapshot({
        state: 'uninitialized',
        initialized: false,
        initialization: null,
        leaseMode: null,
        idleDeadlineMs: null,
      }),
    })),
    run: vi.fn(async () => runReport()),
    synthesize: vi.fn(async () => synthesisReport()),
    transcribeCurrentClip: vi.fn(async () => roundTripReport()),
    runRoundTrip: vi.fn(async () => roundTripReport()),
    cancel: vi.fn(async () => true),
    onEvent: vi.fn((listener) => {
      eventListener = listener;
      return vi.fn();
    }),
  };
  window.speechLab = api;
});

afterEach(() => {
  for (const wrapper of wrappers)
    wrapper.unmount();
  wrappers = [];
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

async function mountApp(): Promise<VueWrapper> {
  const wrapper = mount(App, { attachTo: document.body });
  wrappers.push(wrapper);
  await flushPromises();
  return wrapper;
}

describe('speech dev App', () => {
  it('保留 fixture 测试，默认 batch=1 且只接受正安全整数', async () => {
    const wrapper = await mountApp();
    const batchInput = wrapper.get<HTMLInputElement>('[data-test="batch-size"]');

    expect(batchInput.element.value).toBe('1');
    await wrapper.get('[data-test="run-selected"]').trigger('click');
    await flushPromises();
    expect(api.run).toHaveBeenCalledWith({ fixtureIds: ['fixture.wav'], batchSize: 1 });

    await batchInput.setValue('0');
    expect(wrapper.text()).toContain('批量大小必须是正安全整数');
    expect(wrapper.get<HTMLButtonElement>('[data-test="run-selected"]').element.disabled).toBe(true);
    await batchInput.setValue('3');
    expect(wrapper.text()).toContain('当前机器未验证大于 2 的批量');
  });

  it('接收 fixture 进度并展示批次折算 CPM 与部分结果', async () => {
    const item = succeededItem({ cpmKind: 'batch_derived', cpm: 80 });
    api.run = vi.fn(async () => runReport({
      requestedBatchSize: 2,
      effectiveBatchSize: 2,
      items: [item],
      overallCpm: 80,
      partial: true,
    }));
    window.speechLab = api;
    const wrapper = await mountApp();
    eventListener?.({
      type: 'run-progress',
      runId: 'run-1',
      completed: 1,
      total: 1,
      items: [item],
    });
    await wrapper.get('[data-test="run-selected"]').trigger('click');
    await flushPromises();

    expect(wrapper.text()).toContain('批次折算 CPM');
    expect(wrapper.get('[data-test="benchmark-summary"]').text()).toContain('部分结果');
    expect(wrapper.text()).toContain('80.0');
  });

  it('任务运行中禁用策略切换并展示取消反馈', async () => {
    api.cancel = vi.fn(async () => false);
    const wrapper = await mountApp();
    eventListener?.({
      type: 'snapshot',
      snapshot: labSnapshot({
        operation: 'transcribing',
        asr: providerSnapshot({ operation: 'transcribing', state: 'busy' }),
      }),
    });
    await flushPromises();

    expect(wrapper.get<HTMLSelectElement>('[data-test="runtime-policy"]').element.disabled).toBe(true);
    await wrapper.get('[data-test="cancel-run"]').trigger('click');
    await flushPromises();
    expect(wrapper.get('[data-test="action-error"]').text()).toContain('当前没有可取消的任务');
  });

  it('fixture 播放器使用受限字节，点击时间戳跳转并清理 Blob URL', async () => {
    const wrapper = await mountApp();
    await wrapper.get('[data-test="run-selected"]').trigger('click');
    await flushPromises();
    const audio = wrapper.get<HTMLAudioElement>('.result-panel audio').element;

    await wrapper.get('[data-test="timestamp-row"]').trigger('click');

    expect(api.readFixtureAudio).toHaveBeenCalledWith('fixture.wav');
    expect(audio.currentTime).toBe(1.25);
    wrapper.unmount();
    wrappers = wrappers.filter(candidate => candidate !== wrapper);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:speech-1');
  });

  it('支持单独生成并复用当前 clip 转写，Renderer 不接触路径', async () => {
    const wrapper = await mountApp();
    await wrapper.get('[data-test="tts-voice"]').setValue('Serena');
    await wrapper.get('[data-test="tts-instruction"]').setValue('语速稍慢');

    await wrapper.get('[data-test="synthesize-current"]').trigger('click');
    await flushPromises();

    expect(api.synthesize).toHaveBeenCalledWith({
      text: '银行行长走过人行道。',
      voiceId: 'Serena',
      instruction: '语速稍慢',
    });
    expect(api.readGeneratedAudio).toHaveBeenCalledWith('clip-1');
    expect(wrapper.get<HTMLAudioElement>('[data-test="generated-audio"]').attributes('src')).toBe('blob:speech-2');

    await wrapper.get('[data-test="transcribe-current"]').trigger('click');
    await flushPromises();
    expect(api.transcribeCurrentClip).toHaveBeenCalledOnce();
    expect(wrapper.get('[data-test="round-trip-verdict"]').text()).toBe('一致');
  });

  it('切换策略只提交受支持值', async () => {
    const wrapper = await mountApp();

    await wrapper.get('[data-test="runtime-policy"]').setValue('exclusive');
    await flushPromises();

    expect(api.setRuntimePolicy).toHaveBeenCalledWith('exclusive');
  });

  it('闭环不一致只标记需复核，并支持生成音频时间戳跳转', async () => {
    api.runRoundTrip = vi.fn(async () => roundTripReport({
      transcription: {
        transcript: '银行行长走过人形道。',
        language: 'Chinese',
        timestamps: [{ text: '人形道', start: 0.6, end: 1.1 }],
        normalizedMatch: false,
        similarity: 0.9,
        inferenceSeconds: 1.1,
        failure: null,
      },
      verdict: 'review_required',
    }));
    window.speechLab = api;
    const wrapper = await mountApp();

    await wrapper.get('[data-test="run-round-trip"]').trigger('click');
    await flushPromises();
    await wrapper.get('[data-test="generated-timestamp-row"]').trigger('click');

    expect(wrapper.get('[data-test="round-trip-verdict"]').text()).toBe('不一致需复核');
    expect(wrapper.text()).not.toContain('多音字发音正确');
    expect(wrapper.get<HTMLAudioElement>('[data-test="generated-audio"]').element.currentTime).toBe(0.6);
  });
});
