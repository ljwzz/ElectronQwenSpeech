import assert from 'node:assert/strict';
import test from 'node:test';
import {
  AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS,
} from './index.ts';

const RGB_HEX_PATTERN = /^#[\dA-F]{6}$/u;

test('音频 QA 文本差异色使用可持久化的六位十六进制默认值', () => {
  assert.deepEqual(AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS, {
    referenceText: '#2E7D55',
    recognizedText: '#F72D5B',
  });
  assert.match(AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS.referenceText, RGB_HEX_PATTERN);
  assert.match(AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS.recognizedText, RGB_HEX_PATTERN);
});

test('音频 QA 文本差异色默认值不可变', () => {
  assert.equal(Object.isFrozen(AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS), true);
});
