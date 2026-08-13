<script setup lang="ts">
import type {
  SpeechDevEvent,
  SpeechFixtureSummary,
  SpeechLabSnapshot,
  SpeechRoundTripReport,
  SpeechRunItem,
  SpeechRunReport,
  SpeechRuntimePolicy,
  SpeechSynthesisReport,
  SpeechSynthesisRequest,
} from '../../contracts.ts';

import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import {
  SPEECH_DEV_RUNTIME_CONFIG,
  SPEECH_DEV_TTS_CONFIG,
} from '../../contracts.ts';
import TextDiffView from './TextDiffView.vue';

const emptySnapshot: SpeechLabSnapshot = {
  policy: SPEECH_DEV_RUNTIME_CONFIG.defaultPolicy,
  operation: 'idle',
  activeTaskId: null,
  tts: {
    state: 'uninitialized',
    initialized: false,
    activeTaskId: null,
    initializationSeconds: null,
    initialization: null,
    lastError: null,
  },
  asr: {
    state: 'uninitialized',
    operation: 'idle',
    initialized: false,
    activeTaskId: null,
    leaseMode: null,
    idleDeadlineMs: null,
    initializationSeconds: null,
    initialization: null,
    lastError: null,
  },
  currentClip: null,
  lastError: null,
};

const snapshot = ref<SpeechLabSnapshot>(emptySnapshot);
const fixtures = ref<SpeechFixtureSummary[]>([]);
const selectedIds = ref<string[]>([]);
const previewFixtureId = ref<string | null>(null);
const runItems = ref<SpeechRunItem[]>([]);
const report = ref<SpeechRunReport | null>(null);
const synthesisReport = ref<SpeechSynthesisReport | null>(null);
const roundTripReport = ref<SpeechRoundTripReport | null>(null);
const ttsText = ref('银行行长走过人行道。');
const ttsVoiceId = ref<string>(SPEECH_DEV_TTS_CONFIG.defaultVoiceId);
const ttsInstruction = ref('');
const batchSizeInput = ref('1');
const actionError = ref<string | null>(null);
const loadingFixtures = ref(false);
const running = ref(false);
const nowMs = ref(Date.now());
const fixtureAudioElement = shallowRef<HTMLAudioElement | null>(null);
const generatedAudioElement = shallowRef<HTMLAudioElement | null>(null);
const fixtureAudioUrl = ref<string | null>(null);
const generatedAudioUrl = ref<string | null>(null);
let removeEventListener: (() => void) | undefined;
let countdownTimer: ReturnType<typeof setInterval> | undefined;
let fixtureAudioLoadSequence = 0;
let generatedAudioLoadSequence = 0;

const runnableFixtures = computed(() => fixtures.value.filter(fixture => fixture.error === null));
const selectedRunnableIds = computed(() => selectedIds.value.filter(fixtureId => (
  runnableFixtures.value.some(fixture => fixture.id === fixtureId)
)));
const batchSize = computed(() => {
  if (!/^[1-9]\d*$/u.test(batchSizeInput.value))
    return null;
  const value = Number(batchSizeInput.value);
  return Number.isSafeInteger(value) ? value : null;
});
const batchError = computed(() => batchSize.value === null ? '批量大小必须是正安全整数。' : null);
const unverifiedBatch = computed(() => (batchSize.value ?? 0) > 2);
const busy = computed(() => running.value || snapshot.value.operation !== 'idle');
const currentItem = computed(() => runItems.value.find(item => item.fixtureId === previewFixtureId.value) ?? null);
const hasGeneratedClip = computed(() => snapshot.value.currentClip !== null || synthesisReport.value !== null);
const runtimeSummary = computed(() => {
  const initialization = snapshot.value.tts.initialization ?? snapshot.value.asr.initialization;
  const accelerator = initialization?.accelerator ?? 'metal';
  const acceleratorLabel = accelerator === 'metal' ? 'Metal' : accelerator;
  return `MLX / ${acceleratorLabel} / BF16`;
});
const countdownText = computed(() => {
  const deadline = snapshot.value.asr.idleDeadlineMs;
  if (deadline === null)
    return snapshot.value.asr.initialized ? '计时暂停' : '—';
  const remainingSeconds = Math.max(0, Math.ceil((deadline - nowMs.value) / 1_000));
  const minutes = Math.floor(remainingSeconds / 60);
  const seconds = remainingSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
});

function messageFrom(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatSeconds(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(2)} s`;
}

function formatNumber(value: number | null, digits = 1): string {
  return value === null ? '—' : value.toFixed(digits);
}

function formatSimilarity(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function statusLabel(item: SpeechRunItem): string {
  const labels: Record<SpeechRunItem['status'], string> = {
    pending: '等待',
    running: '运行中',
    succeeded: '成功',
    failed: '失败',
    cancelled: '已取消',
  };
  return labels[item.status];
}

function verdictLabel(reportValue: SpeechRoundTripReport): string {
  const labels: Record<SpeechRoundTripReport['verdict'], string> = {
    consistent: '一致',
    review_required: '不一致需复核',
    validation_failed: '校验失败',
  };
  return labels[reportValue.verdict];
}

function createSynthesisRequest(): SpeechSynthesisRequest {
  const instruction = ttsInstruction.value.trim();
  return {
    text: ttsText.value,
    voiceId: ttsVoiceId.value,
    ...(instruction ? { instruction } : {}),
  };
}

function onSpeechEvent(event: SpeechDevEvent): void {
  if (event.type === 'snapshot') {
    snapshot.value = event.snapshot;
    return;
  }
  runItems.value = event.items;
}

async function loadInitialState(): Promise<void> {
  actionError.value = null;
  try {
    const [nextSnapshot, nextFixtures] = await Promise.all([
      window.speechLab.getSnapshot(),
      window.speechLab.listFixtures(),
    ]);
    snapshot.value = nextSnapshot;
    fixtures.value = nextFixtures;
    const firstRunnable = nextFixtures.find(fixture => fixture.error === null);
    if (firstRunnable) {
      selectedIds.value = [firstRunnable.id];
      previewFixtureId.value = firstRunnable.id;
    }
  } catch (error) {
    actionError.value = messageFrom(error);
  }
}

async function refreshFixtures(): Promise<void> {
  loadingFixtures.value = true;
  actionError.value = null;
  try {
    fixtures.value = await window.speechLab.refreshFixtures();
    const availableIds = new Set(fixtures.value.filter(fixture => fixture.error === null).map(fixture => fixture.id));
    selectedIds.value = selectedIds.value.filter(fixtureId => availableIds.has(fixtureId));
    if (!previewFixtureId.value || !fixtures.value.some(fixture => fixture.id === previewFixtureId.value))
      previewFixtureId.value = fixtures.value[0]?.id ?? null;
  } catch (error) {
    actionError.value = messageFrom(error);
  } finally {
    loadingFixtures.value = false;
  }
}

async function initializeManually(): Promise<void> {
  actionError.value = null;
  running.value = true;
  try {
    snapshot.value = await window.speechLab.initialize();
  } catch (error) {
    actionError.value = messageFrom(error);
    snapshot.value = await window.speechLab.getSnapshot();
  } finally {
    running.value = false;
  }
}

async function stopProviders(): Promise<void> {
  actionError.value = null;
  try {
    snapshot.value = await window.speechLab.stop();
  } catch (error) {
    actionError.value = messageFrom(error);
    snapshot.value = await window.speechLab.getSnapshot();
  }
}

async function setRuntimePolicy(policy: SpeechRuntimePolicy): Promise<void> {
  actionError.value = null;
  running.value = true;
  try {
    snapshot.value = await window.speechLab.setRuntimePolicy(policy);
  } catch (error) {
    actionError.value = messageFrom(error);
    snapshot.value = await window.speechLab.getSnapshot();
  } finally {
    running.value = false;
  }
}

async function loadGeneratedAudio(clipId: string): Promise<void> {
  const sequence = ++generatedAudioLoadSequence;
  if (generatedAudioUrl.value) {
    URL.revokeObjectURL(generatedAudioUrl.value);
    generatedAudioUrl.value = null;
  }
  const audio = await window.speechLab.readGeneratedAudio(clipId);
  if (sequence !== generatedAudioLoadSequence)
    return;
  const bytes = new Uint8Array(audio.data);
  generatedAudioUrl.value = URL.createObjectURL(new Blob([bytes.buffer], { type: audio.mimeType }));
}

async function synthesizeOnly(): Promise<void> {
  actionError.value = null;
  roundTripReport.value = null;
  running.value = true;
  try {
    const result = await window.speechLab.synthesize(createSynthesisRequest());
    synthesisReport.value = result;
    await loadGeneratedAudio(result.clipId);
  } catch (error) {
    actionError.value = messageFrom(error);
  } finally {
    running.value = false;
  }
}

async function transcribeCurrentClip(): Promise<void> {
  actionError.value = null;
  running.value = true;
  try {
    const result = await window.speechLab.transcribeCurrentClip();
    synthesisReport.value = result.synthesis;
    roundTripReport.value = result;
    if (!generatedAudioUrl.value)
      await loadGeneratedAudio(result.synthesis.clipId);
  } catch (error) {
    actionError.value = messageFrom(error);
  } finally {
    running.value = false;
  }
}

async function runRoundTrip(): Promise<void> {
  actionError.value = null;
  roundTripReport.value = null;
  running.value = true;
  try {
    const result = await window.speechLab.runRoundTrip(createSynthesisRequest());
    synthesisReport.value = result.synthesis;
    roundTripReport.value = result;
    await loadGeneratedAudio(result.synthesis.clipId);
  } catch (error) {
    actionError.value = messageFrom(error);
  } finally {
    running.value = false;
  }
}

async function runFixtureIds(fixtureIds: string[]): Promise<void> {
  actionError.value = null;
  report.value = null;
  runItems.value = [];
  if (batchSize.value === null) {
    actionError.value = batchError.value;
    return;
  }
  if (fixtureIds.length === 0) {
    actionError.value = '没有可运行的 fixture。';
    return;
  }
  running.value = true;
  try {
    report.value = await window.speechLab.run({
      fixtureIds,
      batchSize: batchSize.value,
    });
    runItems.value = report.value.items;
  } catch (error) {
    actionError.value = messageFrom(error);
    snapshot.value = await window.speechLab.getSnapshot();
  } finally {
    running.value = false;
  }
}

async function cancelRun(): Promise<void> {
  actionError.value = null;
  try {
    const accepted = await window.speechLab.cancel();
    if (!accepted)
      actionError.value = '当前没有可取消的任务。';
  } catch (error) {
    actionError.value = messageFrom(error);
  }
}

function toggleSelection(fixtureId: string, checked: boolean): void {
  if (checked) {
    if (!selectedIds.value.includes(fixtureId))
      selectedIds.value = [...selectedIds.value, fixtureId];
    return;
  }
  selectedIds.value = selectedIds.value.filter(id => id !== fixtureId);
}

async function loadFixtureAudio(fixtureId: string | null): Promise<void> {
  const sequence = ++fixtureAudioLoadSequence;
  if (fixtureAudioUrl.value) {
    URL.revokeObjectURL(fixtureAudioUrl.value);
    fixtureAudioUrl.value = null;
  }
  if (!fixtureId)
    return;
  try {
    const audio = await window.speechLab.readFixtureAudio(fixtureId);
    if (sequence !== fixtureAudioLoadSequence)
      return;
    const bytes = new Uint8Array(audio.data);
    fixtureAudioUrl.value = URL.createObjectURL(new Blob([bytes.buffer], { type: audio.mimeType }));
  } catch (error) {
    actionError.value = messageFrom(error);
  }
}

function seekFixture(start: number): void {
  if (fixtureAudioElement.value)
    fixtureAudioElement.value.currentTime = start;
}

function seekGenerated(start: number): void {
  if (generatedAudioElement.value)
    generatedAudioElement.value.currentTime = start;
}

watch(previewFixtureId, fixtureId => void loadFixtureAudio(fixtureId));

onMounted(async () => {
  removeEventListener = window.speechLab.onEvent(onSpeechEvent);
  countdownTimer = setInterval(() => {
    nowMs.value = Date.now();
  }, 1_000);
  await loadInitialState();
  await nextTick();
});

onBeforeUnmount(() => {
  fixtureAudioLoadSequence += 1;
  generatedAudioLoadSequence += 1;
  removeEventListener?.();
  if (countdownTimer)
    clearInterval(countdownTimer);
  if (fixtureAudioUrl.value)
    URL.revokeObjectURL(fixtureAudioUrl.value);
  if (generatedAudioUrl.value)
    URL.revokeObjectURL(generatedAudioUrl.value);
});
</script>

<template>
  <main class="speech-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">
          ELECTRON QWEN SPEECH / DEVELOPMENT ONLY
        </p>
        <h1>Qwen 本地语音诊断台</h1>
        <p class="subtitle">
          文本 → TTS → WAV → ASR → 文本 · {{ runtimeSummary }}
        </p>
      </div>
      <div class="topbar-actions">
        <button class="button secondary" data-test="manual-initialize" :disabled="busy" @click="initializeManually">
          初始化 Provider
        </button>
        <button class="button danger" data-test="stop-provider" @click="stopProviders">
          停用
        </button>
      </div>
    </header>

    <section class="status-grid" aria-label="Provider 状态">
      <article class="status-card policy-card">
        <span>模型驻留策略</span>
        <select
          data-test="runtime-policy"
          :value="snapshot.policy"
          :disabled="busy"
          @change="setRuntimePolicy(($event.target as HTMLSelectElement).value as SpeechRuntimePolicy)"
        >
          <option value="resident">
            常驻 resident
          </option>
          <option value="exclusive">
            互斥 exclusive
          </option>
        </select>
        <small>{{ snapshot.operation }}</small>
      </article>
      <article class="status-card">
        <span>TTS Provider</span>
        <strong :class="`state-${snapshot.tts.state}`">{{ snapshot.tts.state }}</strong>
        <small>{{ formatSeconds(snapshot.tts.initializationSeconds) }}</small>
      </article>
      <article class="status-card">
        <span>TTS 运行时 / 加速器</span>
        <strong>{{ snapshot.tts.initialization ? `${snapshot.tts.initialization.runtime} / ${snapshot.tts.initialization.accelerator}` : '—' }}</strong>
        <small>{{ snapshot.tts.initialization ? `${snapshot.tts.initialization.device} · ${snapshot.tts.initialization.modelPrecision} / tokenizer ${snapshot.tts.initialization.speechTokenizerPrecision}` : '—' }}</small>
      </article>
      <article class="status-card">
        <span>ASR Provider</span>
        <strong :class="`state-${snapshot.asr.state}`">{{ snapshot.asr.state }}</strong>
        <small>{{ formatSeconds(snapshot.asr.initializationSeconds) }}</small>
      </article>
      <article class="status-card">
        <span>ASR / Aligner 运行时</span>
        <strong>{{ snapshot.asr.initialization ? `${snapshot.asr.initialization.runtime} / ${snapshot.asr.initialization.accelerator}` : '—' }}</strong>
        <small>{{ snapshot.asr.initialization ? `${snapshot.asr.initialization.device} · ${snapshot.asr.initialization.asrModelPrecision} / ${snapshot.asr.initialization.alignerModelPrecision}` : '—' }}</small>
      </article>
      <article class="status-card">
        <span>ASR 租约 / 倒计时</span>
        <strong>{{ snapshot.asr.leaseMode ?? '—' }}</strong>
        <small data-test="countdown">{{ countdownText }}</small>
      </article>
    </section>

    <div
      v-if="actionError || snapshot.lastError || snapshot.tts.lastError || snapshot.asr.lastError"
      class="error-banner"
      role="alert"
      data-test="action-error"
    >
      {{ actionError ?? `${(snapshot.lastError ?? snapshot.tts.lastError ?? snapshot.asr.lastError)?.code}: ${(snapshot.lastError ?? snapshot.tts.lastError ?? snapshot.asr.lastError)?.message}` }}
    </div>

    <section class="tts-lab">
      <div class="section-heading">
        <div>
          <p class="eyebrow">
            TTS → ASR ROUND TRIP
          </p>
          <h2>中文合成与回转校验</h2>
        </div>
        <span v-if="roundTripReport" class="verdict-badge" :class="`verdict-${roundTripReport.verdict}`" data-test="round-trip-verdict">
          {{ verdictLabel(roundTripReport) }}
        </span>
      </div>

      <div class="tts-layout">
        <div class="tts-controls">
          <label class="field-group tts-text-field">
            <span>原文</span>
            <textarea v-model="ttsText" data-test="tts-text" rows="4" :disabled="busy" />
          </label>
          <div class="tts-options">
            <label class="field-group">
              <span>中文原生音色</span>
              <select v-model="ttsVoiceId" data-test="tts-voice" :disabled="busy">
                <option v-for="voiceId in SPEECH_DEV_TTS_CONFIG.voiceIds" :key="voiceId" :value="voiceId">
                  {{ voiceId }}
                </option>
              </select>
            </label>
            <label class="field-group instruction-field">
              <span>朗读指令（可选）</span>
              <input v-model="ttsInstruction" data-test="tts-instruction" :disabled="busy" placeholder="例如：语速稍慢，清晰朗读">
            </label>
          </div>
          <div class="run-actions tts-actions">
            <button class="button secondary" data-test="synthesize-current" :disabled="busy || !ttsText.trim()" @click="synthesizeOnly">
              仅生成音频
            </button>
            <button class="button secondary" data-test="transcribe-current" :disabled="busy || !hasGeneratedClip" @click="transcribeCurrentClip">
              转写当前音频
            </button>
            <button class="button primary" data-test="run-round-trip" :disabled="busy || !ttsText.trim()" @click="runRoundTrip">
              一键 TTS → ASR
            </button>
            <button class="button danger" data-test="cancel-speech-task" :disabled="!busy" @click="cancelRun">
              取消
            </button>
          </div>
        </div>

        <div class="generated-result">
          <audio
            ref="generatedAudioElement"
            class="audio-player"
            data-test="generated-audio"
            controls
            :src="generatedAudioUrl ?? undefined"
          />
          <div v-if="synthesisReport" class="metric-row generated-metrics">
            <div><span>音频时长</span><strong>{{ formatSeconds(synthesisReport.durationSeconds) }}</strong></div>
            <div><span>TTS 推理</span><strong>{{ formatSeconds(synthesisReport.generationSeconds) }}</strong></div>
            <div><span>TTS 墙钟</span><strong>{{ formatSeconds(synthesisReport.wallSeconds) }}</strong></div>
            <div><span>采样率</span><strong>{{ synthesisReport.sampleRate }} Hz</strong></div>
          </div>
          <div v-if="roundTripReport" class="round-trip-result">
            <div class="metric-row generated-metrics">
              <div><span>识别语言</span><strong>{{ roundTripReport.transcription.language ?? '—' }}</strong></div>
              <div><span>对齐状态</span><strong>{{ roundTripReport.transcription.alignmentStatus ?? '—' }}</strong></div>
              <div><span>规范化一致</span><strong>{{ roundTripReport.transcription.normalizedMatch === null ? '—' : (roundTripReport.transcription.normalizedMatch ? '是' : '否') }}</strong></div>
              <div><span>编辑距离相似度</span><strong>{{ formatSimilarity(roundTripReport.transcription.similarity) }}</strong></div>
              <div><span>ASR 推理</span><strong>{{ formatSeconds(roundTripReport.transcription.inferenceSeconds) }}</strong></div>
              <div><span>闭环墙钟</span><strong>{{ formatSeconds(roundTripReport.wallSeconds) }}</strong></div>
            </div>
            <TextDiffView
              :reference-text="roundTripReport.synthesis.sourceText"
              :recognized-text="roundTripReport.transcription.transcript"
            />
            <p v-if="roundTripReport.transcription.failure" class="inline-error" data-test="round-trip-failure">
              {{ roundTripReport.transcription.failure.stage ?? 'unknown' }} · {{ roundTripReport.transcription.failure.code }} · {{ roundTripReport.transcription.failure.message }}
            </p>
            <details class="timestamps" :open="roundTripReport.transcription.timestamps.length > 0">
              <summary>时间戳（{{ roundTripReport.transcription.timestamps.length }}）</summary>
              <button
                v-for="(timestamp, index) in roundTripReport.transcription.timestamps"
                :key="`${timestamp.start}-${index}`"
                class="timestamp-row"
                data-test="generated-timestamp-row"
                @click="seekGenerated(timestamp.start)"
              >
                <span>{{ timestamp.start.toFixed(2) }} → {{ timestamp.end.toFixed(2) }}</span>
                <strong>{{ timestamp.text }}</strong>
              </button>
            </details>
          </div>
          <div v-else-if="!synthesisReport" class="empty-result compact">
            <strong>等待生成</strong>
            <p>可单独生成 WAV、复用当前音频转写，或直接运行闭环。</p>
          </div>
        </div>
      </div>
    </section>

    <section class="workspace-grid">
      <aside class="fixture-panel">
        <div class="section-heading">
          <div>
            <p class="eyebrow">
              ASR FIXTURE MATRIX
            </p>
            <h2>现有 ASR Fixtures</h2>
          </div>
          <button class="text-button" :disabled="loadingFixtures || busy" @click="refreshFixtures">
            {{ loadingFixtures ? '刷新中…' : '刷新 fixtures' }}
          </button>
        </div>

        <div class="batch-control">
          <label for="batch-size">批量大小</label>
          <input id="batch-size" v-model.trim="batchSizeInput" data-test="batch-size" inputmode="numeric">
          <small v-if="batchError" class="field-error">{{ batchError }}</small>
          <small v-else-if="unverifiedBatch" class="field-warning">当前机器未验证大于 2 的批量。</small>
        </div>

        <div class="run-actions">
          <button
            class="button primary"
            data-test="run-selected"
            :disabled="busy || selectedRunnableIds.length === 0 || batchSize === null"
            @click="runFixtureIds(selectedRunnableIds)"
          >
            运行选中项
          </button>
          <button
            class="button secondary"
            data-test="run-all"
            :disabled="busy || runnableFixtures.length === 0 || batchSize === null"
            @click="runFixtureIds(runnableFixtures.map(fixture => fixture.id))"
          >
            运行全部
          </button>
          <button class="button danger" data-test="cancel-run" :disabled="!busy" @click="cancelRun">
            取消
          </button>
        </div>

        <div class="fixture-list">
          <article
            v-for="fixture in fixtures"
            :key="fixture.id"
            class="fixture-row"
            :class="{ active: previewFixtureId === fixture.id, invalid: fixture.error }"
          >
            <label class="fixture-select">
              <input
                type="checkbox"
                :checked="selectedIds.includes(fixture.id)"
                :disabled="fixture.error !== null"
                @change="toggleSelection(fixture.id, ($event.target as HTMLInputElement).checked)"
              >
              <span />
            </label>
            <button class="fixture-main" @click="previewFixtureId = fixture.id">
              <strong>{{ fixture.title ?? fixture.audioFileName }}</strong>
              <small>{{ fixture.audioFileName }}</small>
              <em v-if="fixture.error">{{ fixture.error }}</em>
            </button>
            <span class="format-badge">{{ fixture.format }}</span>
          </article>
          <p v-if="fixtures.length === 0" class="empty-state">
            没有同名 JSON + WAV/MP3 配对。
          </p>
        </div>
      </aside>

      <section class="result-panel">
        <div class="section-heading">
          <div>
            <p class="eyebrow">
              SELECTED FIXTURE
            </p>
            <h2>{{ fixtures.find(fixture => fixture.id === previewFixtureId)?.title ?? '选择一个 fixture' }}</h2>
          </div>
          <span v-if="currentItem" class="result-status" :class="`result-${currentItem.status}`">
            {{ statusLabel(currentItem) }}
          </span>
        </div>

        <audio ref="fixtureAudioElement" class="audio-player" controls :src="fixtureAudioUrl ?? undefined" />

        <div v-if="report" class="benchmark-strip" :class="{ partial: report.partial }" data-test="benchmark-summary">
          <div><span>完整墙钟</span><strong>{{ formatSeconds(report.wallSeconds) }}</strong></div>
          <div><span>推理总计</span><strong>{{ formatSeconds(report.inferenceSeconds) }}</strong></div>
          <div><span>整体 CPM</span><strong>{{ formatNumber(report.overallCpm) }}</strong></div>
          <div><span>结果</span><strong>{{ report.partial ? '部分结果' : '完整' }}</strong></div>
        </div>

        <article v-if="currentItem" class="transcript-card">
          <div class="metric-row">
            <div><span>语言</span><strong>{{ currentItem.language ?? '—' }}</strong></div>
            <div><span>对齐状态</span><strong>{{ currentItem.alignmentStatus ?? '—' }}</strong></div>
            <div><span>相似度</span><strong>{{ formatSimilarity(currentItem.similarity) }}</strong></div>
            <div><span>字符数</span><strong>{{ currentItem.characterCount }}</strong></div>
            <div><span>推理耗时</span><strong>{{ formatSeconds(currentItem.inferenceSeconds) }}</strong></div>
            <div>
              <span>{{ currentItem.cpmKind === 'batch_derived' ? '批次折算 CPM' : 'CPM' }}</span>
              <strong>{{ formatNumber(currentItem.cpm) }}</strong>
            </div>
          </div>
          <TextDiffView
            :reference-text="currentItem.referenceText"
            :recognized-text="currentItem.transcript"
          />
          <p v-if="currentItem.error" class="inline-error">
            {{ currentItem.error.code }} · {{ currentItem.error.message }}
          </p>
          <details class="timestamps" :open="currentItem.timestamps.length > 0">
            <summary>时间戳（{{ currentItem.timestamps.length }}）</summary>
            <button
              v-for="(timestamp, index) in currentItem.timestamps"
              :key="`${timestamp.start}-${index}`"
              class="timestamp-row"
              data-test="timestamp-row"
              @click="seekFixture(timestamp.start)"
            >
              <span>{{ timestamp.start.toFixed(2) }} → {{ timestamp.end.toFixed(2) }}</span>
              <strong>{{ timestamp.text }}</strong>
            </button>
          </details>
        </article>

        <div v-else class="empty-result">
          <strong>等待运行</strong>
          <p>选择 fixture 后可试听；运行后在此查看全文、指标和逐字时间戳。</p>
        </div>

        <section v-if="runItems.length > 0" class="run-queue">
          <h3>本次运行</h3>
          <button
            v-for="item in runItems"
            :key="item.fixtureId"
            :class="{ active: previewFixtureId === item.fixtureId }"
            @click="previewFixtureId = item.fixtureId"
          >
            <span>{{ item.audioFileName }}</span>
            <strong>{{ statusLabel(item) }}</strong>
          </button>
        </section>
      </section>
    </section>
  </main>
</template>
