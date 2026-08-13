// @vitest-environment node

import type { TTSInitializeResult, TTSSynthesisResult } from '@electron-qwen-speech/application';

import { access, readFile } from 'node:fs/promises';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { QwenLocalTTSProvider } from './qwenLocalTTSProvider.ts';

const TTS_MODEL_PATH = process.env.ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH;
if (!TTS_MODEL_PATH)
  throw new Error('ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH 未配置。');

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

describe.sequential('QwenLocalTTSProvider real MPS integration', () => {
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

  it('在 mps:0 以 bfloat16 初始化 CustomVoice capability', () => {
    expect(initialization).toMatchObject({
      state: 'ready',
      modelPath: TTS_MODEL_PATH,
      device: 'mps:0',
      dtype: 'bfloat16',
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
    expect(wave.toString('ascii', 0, 4)).toBe('RIFF');
    expect(wave.toString('ascii', 8, 12)).toBe('WAVE');
    expect(wave.readUInt16LE(20)).toBe(1);
    expect(wave.readUInt16LE(34)).toBe(16);
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
