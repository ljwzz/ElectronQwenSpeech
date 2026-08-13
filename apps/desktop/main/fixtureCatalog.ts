import type { Dirent } from 'node:fs';
import type { SpeechFixtureAudio, SpeechFixtureSummary } from '../contracts.ts';

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { findDevelopmentRepositoryRoot } from './developmentRepositoryRoot.ts';

const REPOSITORY_ROOT = findDevelopmentRepositoryRoot();
export const SPEECH_FIXTURE_DIRECTORY = path.join(
  REPOSITORY_ROOT,
  'services',
  'asr-sidecar',
  'tests',
  'fixtures',
);
const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav']);

export interface RunnableSpeechFixture extends SpeechFixtureSummary {
  audioPath: string;
  title: string;
  referenceText: string;
  error: null;
}

interface RegisteredSpeechFixture {
  summary: SpeechFixtureSummary;
  audioPath: string;
}

function validateExpectation(value: unknown): { title: string; content: string } {
  if (typeof value !== 'object' || value === null)
    throw new Error('JSON 根节点必须是对象。');
  const record = value as Record<string, unknown>;
  if (typeof record.title !== 'string' || !record.title.trim())
    throw new Error('title 必须是非空字符串。');
  if (typeof record.content !== 'string' || !record.content.trim())
    throw new Error('content 必须是非空字符串。');
  return { title: record.title, content: record.content };
}

export class SpeechFixtureCatalog {
  readonly #directory: string;
  #fixtures = new Map<string, RegisteredSpeechFixture>();

  constructor(directory = SPEECH_FIXTURE_DIRECTORY) {
    this.#directory = directory;
  }

  async refresh(): Promise<SpeechFixtureSummary[]> {
    let entries: Dirent<string>[];
    try {
      entries = await readdir(this.#directory, { withFileTypes: true, encoding: 'utf8' });
    } catch (error) {
      if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
        this.#fixtures = new Map();
        return [];
      }
      throw error;
    }

    const fileNames = entries
      .filter(entry => entry.isFile())
      .map(entry => entry.name)
      .sort((left, right) => left.localeCompare(right, 'en'));
    const jsonByBaseName = new Map<string, string>();
    for (const fileName of fileNames) {
      const extension = path.extname(fileName);
      if (extension.toLowerCase() === '.json')
        jsonByBaseName.set(path.basename(fileName, extension), fileName);
    }

    const registered = new Map<string, RegisteredSpeechFixture>();
    for (const audioFileName of fileNames) {
      const extension = path.extname(audioFileName);
      if (!AUDIO_EXTENSIONS.has(extension.toLowerCase()))
        continue;
      const baseName = path.basename(audioFileName, extension);
      const jsonFileName = jsonByBaseName.get(baseName);
      if (!jsonFileName)
        continue;

      const format = extension.toLowerCase().slice(1) as 'mp3' | 'wav';
      const audioPath = path.join(this.#directory, audioFileName);
      let summary: SpeechFixtureSummary;
      try {
        const expectation = validateExpectation(JSON.parse(
          await readFile(path.join(this.#directory, jsonFileName), 'utf8'),
        ));
        summary = {
          id: audioFileName,
          audioFileName,
          format,
          title: expectation.title,
          referenceText: expectation.content,
          error: null,
        };
      } catch (error) {
        summary = {
          id: audioFileName,
          audioFileName,
          format,
          title: null,
          referenceText: null,
          error: error instanceof Error ? error.message : 'JSON 无法解析。',
        };
      }
      registered.set(summary.id, { summary, audioPath });
    }
    this.#fixtures = registered;
    return this.list();
  }

  list(): SpeechFixtureSummary[] {
    return [...this.#fixtures.values()].map(fixture => ({ ...fixture.summary }));
  }

  resolveRunnable(fixtureIds: string[]): RunnableSpeechFixture[] {
    return fixtureIds.map((fixtureId) => {
      const fixture = this.#requireFixture(fixtureId);
      if (fixture.summary.error || !fixture.summary.title || !fixture.summary.referenceText)
        throw new Error(`Fixture ${fixtureId} 无效：${fixture.summary.error ?? '缺少测试期望。'}`);
      return {
        ...fixture.summary,
        audioPath: fixture.audioPath,
        title: fixture.summary.title,
        referenceText: fixture.summary.referenceText,
        error: null,
      };
    });
  }

  async readAudio(fixtureId: string): Promise<SpeechFixtureAudio> {
    const fixture = this.#requireFixture(fixtureId);
    const buffer = await readFile(fixture.audioPath);
    return {
      id: fixtureId,
      mimeType: fixture.summary.format === 'wav' ? 'audio/wav' : 'audio/mpeg',
      data: Uint8Array.from(buffer),
    };
  }

  #requireFixture(fixtureId: string): RegisteredSpeechFixture {
    if (typeof fixtureId !== 'string' || !fixtureId.trim())
      throw new Error('fixtureId 必须是非空字符串。');
    const fixture = this.#fixtures.get(fixtureId);
    if (!fixture)
      throw new Error(`未知 fixtureId：${fixtureId}`);
    return fixture;
  }
}
