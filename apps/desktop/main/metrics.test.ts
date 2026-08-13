// @vitest-environment node

import { describe, expect, it } from 'vitest';
import {
  calculateCpm,
  calculateSpeechSimilarity,
  countSpeechCharacters,
  normalizeSpeechText,
} from './metrics.ts';

describe('speech metrics', () => {
  it('以 NFKC、小写和 Unicode 字母数字 code point 归一化', () => {
    expect(normalizeSpeechText('Ａb，你好！１２')).toBe('ab你好12');
    expect(countSpeechCharacters('Ａb，你好！１２')).toBe(6);
  });

  it('按归一化 Levenshtein 距离计算相似度', () => {
    expect(calculateSpeechSimilarity('你好，Test!', '你好test')).toBe(1);
    expect(calculateSpeechSimilarity('abc', 'adc')).toBeCloseTo(2 / 3);
    expect(calculateSpeechSimilarity('', '')).toBe(1);
  });

  it('只对正推理耗时计算 CPM', () => {
    expect(calculateCpm(120, 30)).toBe(240);
    expect(calculateCpm(10, 0)).toBeNull();
  });
});
