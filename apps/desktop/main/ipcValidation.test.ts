// @vitest-environment node

import { describe, expect, it } from 'vitest';
import {
  requireFixtureId,
  requireNoArgument,
  requireRunRequest,
  requireRuntimePolicy,
  requireSynthesisRequest,
} from './ipcValidation.ts';

describe('speech dev IPC validation', () => {
  it('只接受无参数调用和非空 opaque ID', () => {
    expect(() => requireNoArgument(undefined)).not.toThrow();
    expect(() => requireNoArgument(null)).toThrow('不接受参数');
    expect(requireFixtureId('clip-1')).toBe('clip-1');
    expect(() => requireFixtureId('../audio.wav')).not.toThrow();
    expect(() => requireFixtureId('')).toThrow('非空字符串');
  });

  it('只接受 resident 与 exclusive 策略', () => {
    expect(requireRuntimePolicy('resident')).toBe('resident');
    expect(requireRuntimePolicy('exclusive')).toBe('exclusive');
    expect(() => requireRuntimePolicy('cpu')).toThrow('resident 或 exclusive');
  });

  it('严格校验 TTS 请求字段与类型', () => {
    expect(requireSynthesisRequest({
      text: '银行行长走过人行道。',
      voiceId: 'Vivian',
      instruction: '清晰朗读',
    })).toEqual({
      text: '银行行长走过人行道。',
      voiceId: 'Vivian',
      instruction: '清晰朗读',
    });
    expect(() => requireSynthesisRequest({ text: '', voiceId: 'Vivian' })).toThrow('text');
    expect(() => requireSynthesisRequest({ text: '文本', voiceId: 'Vivian', path: '/tmp/a.wav' }))
      .toThrow('未知字段');
    expect(() => requireSynthesisRequest({ text: '文本', voiceId: 'Vivian', instruction: 1 }))
      .toThrow('instruction');
  });

  it('严格校验 fixture 列表、重复项和 batchSize', () => {
    expect(requireRunRequest({ fixtureIds: ['one', 'two'], batchSize: 2 })).toEqual({
      fixtureIds: ['one', 'two'],
      batchSize: 2,
    });
    expect(() => requireRunRequest({ fixtureIds: ['one', 'one'], batchSize: 1 })).toThrow('不能重复');
    expect(() => requireRunRequest({ fixtureIds: ['one'], batchSize: 0 })).toThrow('正安全整数');
    expect(() => requireRunRequest({ fixtureIds: ['one'], batchSize: 1, extra: true })).toThrow('未知字段');
  });
});
