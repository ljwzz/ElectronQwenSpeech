// @vitest-environment node

import type {
  ASRBatchTranscriptionItemResult,
  ASRInitializeResult,
  ASRTranscriptionResult,
} from '@electron-qwen-speech/application';

import { readdirSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { QwenLocalASRProvider } from './qwenLocalASRProvider.ts';

const REPOSITORY_ROOT = fileURLToPath(new URL('../../../../', import.meta.url));
const ASR_FIXTURE_DIRECTORY = path.join(
  REPOSITORY_ROOT,
  'services',
  'asr-sidecar',
  'tests',
  'fixtures',
);
const AUDIO_FIXTURE_EXTENSIONS = new Set(['.mp3', '.wav']);
const MIN_ASR_FIXTURE_TRANSCRIPT_SIMILARITY = 0.7;
const ALIGNER_MODEL_PATH = process.env.ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH;
const ASR_MODEL_PATH = process.env.ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH;
if (!ALIGNER_MODEL_PATH || !ASR_MODEL_PATH)
  throw new Error('ASR 与 Forced Aligner 模型路径环境变量未配置。');

interface ASRFixtureCase {
  taskId: string;
  audioFileName: string;
  audioPath: string;
  expectationPath: string;
}

interface ASRFixtureExpectation {
  title: string;
  content: string;
}

function discoverASRFixtureCases(): ASRFixtureCase[] {
  const fileNames = readdirSync(ASR_FIXTURE_DIRECTORY, { withFileTypes: true })
    .filter(entry => entry.isFile())
    .map(entry => entry.name)
    .sort();
  const expectationFileNameByBaseName = new Map<string, string>();
  for (const fileName of fileNames) {
    const extension = path.extname(fileName);
    if (extension.toLowerCase() === '.json')
      expectationFileNameByBaseName.set(path.basename(fileName, extension), fileName);
  }

  const cases = fileNames
    .filter((fileName) => {
      const extension = path.extname(fileName);
      return AUDIO_FIXTURE_EXTENSIONS.has(extension.toLowerCase())
        && expectationFileNameByBaseName.has(path.basename(fileName, extension));
    })
    .map((audioFileName, index) => {
      const extension = path.extname(audioFileName);
      const baseName = path.basename(audioFileName, extension);
      const expectationFileName = expectationFileNameByBaseName.get(baseName)!;
      return {
        taskId: `real-fixture-${index + 1}`,
        audioFileName,
        audioPath: path.join(ASR_FIXTURE_DIRECTORY, audioFileName),
        expectationPath: path.join(ASR_FIXTURE_DIRECTORY, expectationFileName),
      };
    });
  if (cases.length === 0)
    throw new Error(`ASR fixture 目录中不存在同名 JSON 与音频文件：${ASR_FIXTURE_DIRECTORY}`);
  return cases;
}

const ASR_FIXTURE_CASES = discoverASRFixtureCases();
const CANCELLATION_FIXTURE_PATH = ASR_FIXTURE_CASES[0]!.audioPath;

function normalizeTranscript(value: string): string {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]/gu, '');
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

  const editDistance = previousRow[expectedCharacters.length]!;
  return 1 - editDistance / maximumLength;
}

function assertValidTimestamps(result: ASRTranscriptionResult | ASRBatchTranscriptionItemResult): void {
  expect(result.timestamps.length).toBeGreaterThan(0);
  for (const [index, timestamp] of result.timestamps.entries()) {
    expect(timestamp.text.length).toBeGreaterThan(0);
    expect(Number.isFinite(timestamp.start)).toBe(true);
    expect(Number.isFinite(timestamp.end)).toBe(true);
    expect(timestamp.start).toBeGreaterThanOrEqual(0);
    expect(timestamp.end).toBeGreaterThanOrEqual(timestamp.start);
    if (index > 0) {
      const previous = result.timestamps[index - 1]!;
      expect(timestamp.start).toBeGreaterThanOrEqual(previous.start);
      expect(timestamp.end).toBeGreaterThanOrEqual(previous.end);
    }
  }
}

async function loadASRFixtureExpectation(expectationPath: string): Promise<ASRFixtureExpectation> {
  const value: unknown = JSON.parse(await readFile(expectationPath, 'utf8'));
  if (
    typeof value !== 'object'
    || value === null
    || !('title' in value)
    || typeof value.title !== 'string'
    || value.title.length === 0
    || !('content' in value)
    || typeof value.content !== 'string'
    || value.content.length === 0
  ) {
    throw new Error(`ASR fixture 期望文件格式无效：${expectationPath}`);
  }
  return value as ASRFixtureExpectation;
}

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

describe.sequential('QwenLocalASRProvider real MPS integration', () => {
  const stderr: string[] = [];
  let initialization: ASRInitializeResult;
  const provider = new QwenLocalASRProvider({
    timeouts: {
      controlMs: 10_000,
      initializeMs: 600_000,
      transcribeMs: 300_000,
      shutdownMs: 30_000,
    },
    onStderr: output => stderr.push(output),
  });

  beforeAll(async () => {
    try {
      const health = await provider.health();
      const status = await provider.getStatus();
      expect(health).toMatchObject({
        healthy: true,
        state: 'uninitialized',
        initialized: false,
        loadCount: 0,
      });
      expect(status).toMatchObject({
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
      await expect(provider.getStatus()).resolves.toMatchObject({ state: 'disposed' });
    } catch (error) {
      process.stderr.write(stderr.join(''));
      throw error;
    }
  });

  it('在 mps:0 以 bfloat16 初始化 ASR 与 Forced Aligner', async () => {
    expect(initialization).toMatchObject({
      state: 'ready',
      asrModelPath: ASR_MODEL_PATH,
      alignerModelPath: ALIGNER_MODEL_PATH,
      asrDevice: 'mps:0',
      alignerDevice: 'mps:0',
      asrDtype: 'bfloat16',
      alignerDtype: 'bfloat16',
      loadCount: 1,
    });
  }, 600_000);

  it.each(ASR_FIXTURE_CASES)(
    '自动转写 $audioFileName 并匹配同名 JSON',
    async ({ taskId, audioPath, expectationPath }) => {
      const expectation = await loadASRFixtureExpectation(expectationPath);
      const result = await provider.transcribe({
        taskId,
        audioPath,
      });

      expect(result.taskId).toBe(taskId);
      expect(result.language).toEqual(expect.any(String));
      expect(result.language!.length).toBeGreaterThan(0);
      expect(
        calculateTranscriptSimilarity(result.text, expectation.content),
        expectation.title,
      ).toBeGreaterThanOrEqual(MIN_ASR_FIXTURE_TRANSCRIPT_SIMILARITY);
      assertValidTimestamps(result);
    },
    300_000,
  );

  it('使用一次 SDK 推理批量转写两个 fixture 并保持输入顺序', async () => {
    const fixtureCases = ASR_FIXTURE_CASES.slice(0, 2);
    expect(fixtureCases).toHaveLength(2);
    const expectations = await Promise.all(
      fixtureCases.map(fixtureCase => loadASRFixtureExpectation(fixtureCase.expectationPath)),
    );

    const result = await provider.transcribeBatch({
      taskId: 'real-batch-two',
      items: fixtureCases.map(fixtureCase => ({
        itemId: fixtureCase.audioFileName,
        audioPath: fixtureCase.audioPath,
      })),
    });

    expect(result.taskId).toBe('real-batch-two');
    expect(result.items.map(item => item.itemId)).toEqual(
      fixtureCases.map(fixtureCase => fixtureCase.audioFileName),
    );
    for (const [index, item] of result.items.entries()) {
      expect(item.language).toEqual(expect.any(String));
      expect(
        calculateTranscriptSimilarity(item.text, expectations[index]!.content),
        expectations[index]!.title,
      ).toBeGreaterThanOrEqual(MIN_ASR_FIXTURE_TRANSCRIPT_SIMILARITY);
      assertValidTimestamps(item);
    }
  }, 300_000);

  it('运行中任务延迟取消，排队任务直接取消', async () => {
    const running = provider.transcribe({
      taskId: 'real-running-cancel',
      audioPath: CANCELLATION_FIXTURE_PATH,
    });
    const runningCancelled = expect(running).rejects.toMatchObject({ code: 'TASK_CANCELLED' });
    await waitFor(async () => {
      const status = await provider.getStatus();
      return status.state === 'busy' && status.currentTaskId === 'real-running-cancel';
    }, 10_000, '首个真实转写进入运行状态');

    const queued = provider.transcribe({
      taskId: 'real-queued-cancel',
      audioPath: CANCELLATION_FIXTURE_PATH,
    });
    const queuedCancelled = expect(queued).rejects.toMatchObject({ code: 'TASK_CANCELLED' });
    await waitFor(async () => (await provider.getStatus()).queuedTaskCount === 1, 10_000, '第二个真实转写进入队列');

    await expect(provider.cancel('real-running-cancel')).resolves.toEqual({
      taskId: 'real-running-cancel',
      status: 'cancel_requested',
    });
    await expect(provider.cancel('real-queued-cancel')).resolves.toEqual({
      taskId: 'real-queued-cancel',
      status: 'cancelled',
    });
    await runningCancelled;
    await queuedCancelled;
  }, 300_000);

  it('音频不存在时返回结构化错误且 Sidecar 保持响应', async () => {
    await expect(provider.transcribe({
      taskId: 'real-missing-audio',
      audioPath: path.join(REPOSITORY_ROOT, 'does-not-exist.wav'),
    })).rejects.toMatchObject({ code: 'AUDIO_NOT_FOUND' });

    await expect(provider.health()).resolves.toMatchObject({
      healthy: true,
      state: 'ready',
      initialized: true,
    });
  });

  it('重复初始化不重新加载模型', async () => {
    await expect(provider.initialize()).resolves.toMatchObject({
      state: 'ready',
      loadCount: 1,
    });
    await expect(provider.getStatus()).resolves.toMatchObject({
      state: 'ready',
      loadCount: 1,
    });
  });
});
