// @vitest-environment node

import { Buffer } from 'node:buffer';
import { access } from 'node:fs/promises';
import process from 'node:process';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { QWEN_LOCAL_ASR_DEFAULTS, QwenLocalASRProvider } from './asr/qwenLocalASRProvider.ts';
import { SpeechLabSession } from './speechLabSession.ts';
import { QWEN_LOCAL_TTS_DEFAULTS, QwenLocalTTSProvider } from './tts/qwenLocalTTSProvider.ts';

const REAL_REQUEST = {
  text: '银行行长走过人行道。',
  voiceId: 'Vivian',
  instruction: '自然、清晰地朗读',
} as const;

function normalizeTranscript(value: string): string {
  return value.normalize('NFKC').toLowerCase().replace(/[^\p{Letter}\p{Number}]/gu, '');
}

function calculateTranscriptSimilarity(actual: string, expected: string): number {
  const actualCharacters = [...normalizeTranscript(actual)];
  const expectedCharacters = [...normalizeTranscript(expected)];
  const maximumLength = Math.max(actualCharacters.length, expectedCharacters.length);
  if (maximumLength === 0)
    return 1;
  let previousRow = expectedCharacters.map((_, index) => index + 1);
  previousRow.unshift(0);
  for (const [actualIndex, actualCharacter] of actualCharacters.entries()) {
    const currentRow = [actualIndex + 1];
    for (const [expectedIndex, expectedCharacter] of expectedCharacters.entries()) {
      currentRow.push(Math.min(
        currentRow[expectedIndex]! + 1,
        previousRow[expectedIndex + 1]! + 1,
        previousRow[expectedIndex]! + (actualCharacter === expectedCharacter ? 0 : 1),
      ));
    }
    previousRow = currentRow;
  }
  return 1 - previousRow[expectedCharacters.length]! / maximumLength;
}

describe.sequential('SpeechLabSession real MLX round trip', () => {
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

  it('resident 同时保持 TTS 与 ASR 的 MLX/Metal capability', () => {
    const expectedTTS = QWEN_LOCAL_TTS_DEFAULTS;
    const expectedASR = QWEN_LOCAL_ASR_DEFAULTS;
    const snapshot = session.snapshot();
    expect(snapshot).toMatchObject({
      policy: 'resident',
      tts: {
        initialized: true,
        state: 'ready',
        initialization: {
          runtime: 'mlx',
          accelerator: expectedTTS.accelerator,
          device: expectedTTS.device,
          modelPrecision: expectedTTS.modelPrecision,
          speechTokenizerPrecision: expectedTTS.speechTokenizerPrecision,
        },
      },
      asr: {
        initialized: true,
        state: 'ready',
        leaseMode: 'resident',
        initialization: {
          runtime: 'mlx',
          accelerator: expectedASR.accelerator,
          device: expectedASR.device,
          asrModelPrecision: expectedASR.asrModelPrecision,
          alignerModelPrecision: expectedASR.alignerModelPrecision,
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
    expect(calculateTranscriptSimilarity(report.transcription.transcript!, REAL_REQUEST.text)).toBeGreaterThanOrEqual(0.7);
    expect(report.transcription.alignmentStatus).toBe('aligned');
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
    expect(calculateTranscriptSimilarity(report.transcription.transcript!, REAL_REQUEST.text)).toBeGreaterThanOrEqual(0.7);
    expect(report.transcription.alignmentStatus).toBe('aligned');
    expect(['consistent', 'review_required']).toContain(report.verdict);
    expect(session.snapshot()).toMatchObject({
      policy: 'exclusive',
      operation: 'idle',
      tts: { initialized: false, state: 'uninitialized' },
      asr: { initialized: true, state: 'ready', leaseMode: 'manual' },
    });
  }, 900_000);
});
