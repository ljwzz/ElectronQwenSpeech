// @vitest-environment node

import type { TTSInitializeResult, TTSSynthesisResult } from '@electron-qwen-speech/application';

import { Buffer } from 'node:buffer';
import { access, readFile } from 'node:fs/promises';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { QWEN_LOCAL_TTS_DEFAULTS, QwenLocalTTSProvider } from './qwenLocalTTSProvider.ts';

async function waitFor(
  predicate: () => Promise<boolean>,
  timeoutMs: number,
  description: string,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate())
      return;
    await new Promise<void>(resolve => setTimeout(resolve, 25));
  }
  throw new Error(`等待 ${description} 超时。`);
}

describe.sequential('QwenLocalTTSProvider real MLX integration', () => {
  const stderr: string[] = [];
  const generatedPaths: string[] = [];
  const provider = new QwenLocalTTSProvider({
    timeouts: {
      controlMs: 10_000,
      initializeMs: 600_000,
      synthesizeMs: 600_000,
      shutdownMs: 30_000,
    },
    onStderr: output => stderr.push(output),
  });
  let initialization: TTSInitializeResult;
  let synthesis: TTSSynthesisResult;

  function assertValidPcm16Wave(wave: Buffer): void {
    expect(wave.toString('ascii', 0, 4)).toBe('RIFF');
    expect(wave.toString('ascii', 8, 12)).toBe('WAVE');
    expect(wave.readUInt16LE(20)).toBe(1);
    expect(wave.readUInt16LE(34)).toBe(16);
    const dataOffset = wave.indexOf(Buffer.from('data'));
    expect(dataOffset).toBeGreaterThanOrEqual(36);
    const dataSize = wave.readUInt32LE(dataOffset + 4);
    expect(dataSize).toBeGreaterThan(0);
    const samples = new Int16Array(
      wave.buffer,
      wave.byteOffset + dataOffset + 8,
      dataSize / Int16Array.BYTES_PER_ELEMENT,
    );
    expect(samples.length).toBeGreaterThan(0);
    expect(samples.some(sample => sample !== 0)).toBe(true);
  }

  beforeAll(async () => {
    try {
      const health = await provider.health();
      expect(health).toMatchObject({
        healthy: true,
        state: 'uninitialized',
        initialized: false,
        loadCount: 0,
      });
      initialization = await provider.initialize();
    } catch (error) {
      process.stderr.write(stderr.join(''));
      throw error;
    }
  });

  afterAll(async () => {
    try {
      await provider.dispose();
      for (const audioPath of generatedPaths)
        await expect(access(audioPath)).rejects.toBeDefined();
    } catch (error) {
      process.stderr.write(stderr.join(''));
      throw error;
    }
  });

  it('在 Metal gpu:0 初始化固定 MLX CustomVoice capability', () => {
    const expected = QWEN_LOCAL_TTS_DEFAULTS;
    expect(initialization).toMatchObject({
      state: 'ready',
      runtime: 'mlx',
      accelerator: expected.accelerator,
      modelPath: process.env.ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH,
      device: expected.device,
      modelFormat: expected.modelFormat,
      modelRevision: expected.modelRevision,
      modelPrecision: expected.modelPrecision,
      speechTokenizerPrecision: expected.speechTokenizerPrecision,
      loadCount: 1,
    });
    expect(initialization.supportedLanguages.map(value => value.toLowerCase())).toContain('chinese');
    expect(initialization.supportedVoiceIds.map(value => value.toLowerCase())).toContain('vivian');
  }, 600_000);

  it('把多音字文本生成 PCM 16-bit 单声道 WAV', async () => {
    synthesis = await provider.synthesize({
      taskId: 'real-polyphone',
      text: '银行行长走过人行道。',
      language: 'Chinese',
      voiceId: 'Vivian',
      instruction: '自然、清晰地朗读',
    });
    generatedPaths.push(synthesis.audioPath);
    const wave = await readFile(synthesis.audioPath);

    expect(synthesis).toMatchObject({
      mimeType: 'audio/wav',
      sampleRate: 24000,
      channels: 1,
    });
    expect(synthesis.frameCount).toBeGreaterThan(0);
    expect(synthesis.durationSeconds).toBeGreaterThan(0);
    expect(synthesis.generationSeconds).toBeGreaterThan(0);
    assertValidPcm16Wave(wave);
  }, 600_000);

  it.each([
    {
      taskId: 'real-date',
      text: '今天是2026年8月13日，下午3点30分。',
      instruction: '自然、清晰地朗读',
    },
    {
      taskId: 'real-dialogue',
      text: '“别着急，”她轻声说，“我们马上出发。”',
      instruction: '用温柔、克制的语气朗读对白',
    },
  ])('生成并校验 $taskId PCM16 音频', async ({ taskId, text, instruction }) => {
    const result = await provider.synthesize({
      taskId,
      text,
      language: 'Chinese',
      voiceId: 'Vivian',
      instruction,
    });
    generatedPaths.push(result.audioPath);
    expect(result).toMatchObject({ sampleRate: 24_000, channels: 1 });
    expect(result.durationSeconds).toBeGreaterThan(0);
    assertValidPcm16Wave(await readFile(result.audioPath));
  }, 600_000);

  it('运行中取消会丢弃推理结果和输出文件', async () => {
    const taskId = 'real-cancel';
    const pending = provider.synthesize({
      taskId,
      text: '银行行长走过人行道。'.repeat(5),
      language: 'Chinese',
      voiceId: 'Vivian',
    });
    await waitFor(async () => (await provider.getStatus()).currentTaskId === taskId, 30_000, 'TTS 任务进入运行态');
    await expect(provider.cancel(taskId)).resolves.toMatchObject({
      taskId,
      status: 'cancel_requested',
    });
    await expect(pending).rejects.toMatchObject({ code: 'TASK_CANCELLED' });
  }, 600_000);
});
