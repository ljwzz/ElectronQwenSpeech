<script setup lang="ts">
import type { ExactTextDiffSegment } from './textDiff.ts';

import { AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS } from '@electron-qwen-speech/contracts';
import { computed } from 'vue';
import { createExactTextDiff } from './textDiff.ts';

const props = defineProps<{
  referenceText: string;
  recognizedText: string | null;
}>();

const comparison = computed(() => (
  props.recognizedText === null
    ? null
    : createExactTextDiff(props.referenceText, props.recognizedText)
));

const diffColorStyle = {
  '--speech-reference-diff-color': AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS.referenceText,
  '--speech-recognized-diff-color': AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS.recognizedText,
};

function displaySegmentValue(segment: ExactTextDiffSegment): string {
  if (segment.kind === 'line-break')
    return '↵';
  if (!segment.changed)
    return segment.value;
  return segment.value
    .replaceAll('\r', '␍')
    .replaceAll('\t', '→')
    .replaceAll(' ', '·')
    .replaceAll('\u00A0', '⍽');
}
</script>

<template>
  <section
    class="text-diff"
    data-test="text-diff"
    :style="diffColorStyle"
    aria-label="参考全文与识别全文精确差异"
  >
    <div v-if="props.recognizedText === null" class="text-diff-state" data-test="text-diff-pending">
      <strong>尚未产生识别全文</strong>
      <span>转写完成后显示逐行、逐字差异。</span>
    </div>

    <div v-else-if="comparison?.identical" class="text-diff-state text-diff-state--equal" data-test="text-diff-equal">
      <strong>全文一致</strong>
      <span>没有需要显示的差异行。</span>
    </div>

    <ol v-else class="text-diff-hunks" aria-label="全文差异块">
      <li
        v-for="(hunk, hunkIndex) in comparison?.hunks"
        :key="hunk.id"
        class="text-diff-hunk"
        data-test="text-diff-hunk"
      >
        <div class="text-diff-hunk-label">
          差异 {{ hunkIndex + 1 }}
        </div>

        <section class="text-diff-side text-diff-side--reference" data-test="reference-diff-side" aria-label="参考全文差异">
          <header>
            <strong>参考全文</strong>
            <span>仅显示差异行</span>
          </header>
          <div
            v-for="(line, lineIndex) in hunk.referenceLines"
            :key="`reference-${line.lineNumber ?? 'empty'}-${lineIndex}`"
            class="text-diff-line"
          >
            <span class="text-diff-line-kind">参考</span>
            <span class="text-diff-line-number">{{ line.lineNumber ?? '—' }}</span>
            <code>
              <span v-if="line.missing" class="text-diff-missing">∅ 无对应内容</span>
              <span
                v-for="(segment, segmentIndex) in line.segments"
                v-else
                :key="segmentIndex"
                :class="{ 'text-diff-character': segment.changed }"
                :data-test="segment.changed ? 'reference-change' : undefined"
              >{{ displaySegmentValue(segment) }}</span>
            </code>
          </div>
        </section>

        <section class="text-diff-side text-diff-side--recognized" data-test="recognized-diff-side" aria-label="识别全文差异">
          <header>
            <strong>识别全文</strong>
            <span>仅显示差异行</span>
          </header>
          <div
            v-for="(line, lineIndex) in hunk.recognizedLines"
            :key="`recognized-${line.lineNumber ?? 'empty'}-${lineIndex}`"
            class="text-diff-line"
          >
            <span class="text-diff-line-kind">识别</span>
            <span class="text-diff-line-number">{{ line.lineNumber ?? '—' }}</span>
            <code>
              <span v-if="line.missing" class="text-diff-missing">∅ 无对应内容</span>
              <span
                v-for="(segment, segmentIndex) in line.segments"
                v-else
                :key="segmentIndex"
                :class="{ 'text-diff-character': segment.changed }"
                :data-test="segment.changed ? 'recognized-change' : undefined"
              >{{ displaySegmentValue(segment) }}</span>
            </code>
          </div>
        </section>
      </li>
    </ol>
  </section>
</template>
