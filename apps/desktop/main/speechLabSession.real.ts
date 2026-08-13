// @vitest-environment node

import { Buffer } from 'node:buffer';
import { access } from 'node:fs/promises';
import process from 'node:process';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { QwenLocalASRProvider } from './asr/qwenLocalASRProvider.ts';
import { SpeechLabSession } from './speechLabSession.ts';
import { QwenLocalTTSProvider } from './tts/qwenLocalTTSProvider.ts';

const REAL_REQUEST = {
  text: '银行行长走过人行道。',
  voiceId: 'Vivian',
  instruction: '自然、清晰地朗读',
} as const;

describe.sequential('SpeechLabSession real MPS round trip', () => {
  const stderr: string[] = [];
  let outputDirectory = '';
  const session = new SpeechLabSession({
    asrProviderFactory: () => new QwenLocalASRProvider({
      timeouts: {
        controlMs: 10_000,
        initializeMs: 600_000,
        transcribeMs: 600_000,
        shutdownMs: 30_000,
      },
      onStderr: output => stderr.push(output),
    }),
    ttsProviderFactory: (directory) => {
      outputDirectory = directory;
      return new QwenLocalTTSProvider({
        outputDirectory: directory,
        timeouts: {
          controlMs: 10_000,
          initializeMs: 600_000,
          synthesizeMs: 600_000,
          shutdownMs: 30_000,
        },
        onStderr: output => stderr.push(output),
      });
    },
  });

  beforeAll(async () => {
    try {
      await session.initialize();
    } catch (error) {
      process.stderr.write(stderr.join(''));
      throw error;
    }
  }, 900_000);

  afterAll(async () => {
    try {
      await session.dispose();
      await expect(access(outputDirectory)).rejects.toBeDefined();
    } catch (error) {
      process.stderr.write(stderr.join(''));
      throw error;
    }
  }, 120_000);

  it('resident 同时保持 TTS 与 ASR 的 MPS/bfloat16 capability', () => {
    const snapshot = session.snapshot();
    expect(snapshot).toMatchObject({
      policy: 'resident',
      tts: {
        initialized: true,
        state: 'ready',
        initialization: { device: 'mps:0', dtype: 'bfloat16' },
      },
      asr: {
        initialized: true,
        state: 'ready',
        leaseMode: 'resident',
        initialization: {
          asrDevice: 'mps:0',
          alignerDevice: 'mps:0',
          asrDtype: 'bfloat16',
          alignerDtype: 'bfloat16',
        },
      },
    });
  });

  it('resident 完成真实文本到 WAV 再到文本的闭环', async () => {
    const report = await session.runRoundTrip(REAL_REQUEST);
    const audio = await session.readGeneratedAudio(report.synthesis.clipId);

    expect(Buffer.from(audio.data).toString('ascii', 0, 4)).toBe('RIFF');
    expect(report.synthesis).toMatchObject({
      sourceText: REAL_REQUEST.text,
      language: 'Chinese',
      voiceId: 'Vivian',
      sampleRate: 24_000,
      channels: 1,
    });
    expect(report.transcription.failure).toBeNull();
    expect(report.transcription.transcript).toEqual(expect.any(String));
    expect(report.transcription.transcript!.length).toBeGreaterThan(0);
    expect(report.transcription.timestamps.length).toBeGreaterThan(0);
    expect(['consistent', 'review_required']).toContain(report.verdict);
    expect(session.snapshot()).toMatchObject({
      tts: { initialized: true },
      asr: { initialized: true },
    });
  }, 900_000);

  it('exclusive 在 TTS 与 ASR 之间释放并重新加载 Provider', async () => {
    await session.setRuntimePolicy('exclusive');
    const report = await session.runRoundTrip(REAL_REQUEST);

    expect(report.transcription.failure).toBeNull();
    expect(report.transcription.transcript).toEqual(expect.any(String));
    expect(['consistent', 'review_required']).toContain(report.verdict);
    expect(session.snapshot()).toMatchObject({
      policy: 'exclusive',
      operation: 'idle',
      tts: { initialized: false, state: 'uninitialized' },
      asr: { initialized: true, state: 'ready', leaseMode: 'manual' },
    });
  }, 900_000);
});
