// @vitest-environment node

import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { SpeechFixtureCatalog } from './fixtureCatalog.ts';

const temporaryDirectories: string[] = [];

async function createFixtureDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'electron-qwen-speech-fixtures-'));
  temporaryDirectories.push(directory);
  return directory;
}

async function addFile(directory: string, fileName: string, content: string | Uint8Array): Promise<void> {
  await writeFile(path.join(directory, fileName), content);
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map(directory => rm(directory, {
    force: true,
    recursive: true,
  })));
});

describe('speechFixtureCatalog', () => {
  it('只注册完全同名的 JSON 与音频并按音频文件名稳定排序', async () => {
    const directory = await createFixtureDirectory();
    await addFile(directory, 'b.json', '{"title":"B","content":"参考 B"}');
    await addFile(directory, 'b.wav', new Uint8Array([1]));
    await addFile(directory, 'a.json', '{"title":"A","content":"参考 A"}');
    await addFile(directory, 'a.mp3', new Uint8Array([2]));
    await addFile(directory, 'unpaired.wav', new Uint8Array([3]));
    await addFile(directory, 'json-only.json', '{"title":"X","content":"X"}');

    const fixtures = await new SpeechFixtureCatalog(directory).refresh();

    expect(fixtures.map(fixture => fixture.audioFileName)).toEqual(['a.mp3', 'b.wav']);
    expect(fixtures[0]).toMatchObject({
      id: 'a.mp3',
      title: 'A',
      referenceText: '参考 A',
      error: null,
    });
  });

  it('同一 basename 的 WAV 与 MP3 注册为共享 JSON 的两个项目', async () => {
    const directory = await createFixtureDirectory();
    await addFile(directory, 'shared.json', '{"title":"共享","content":"共同参考"}');
    await addFile(directory, 'shared.mp3', new Uint8Array([1, 2]));
    await addFile(directory, 'shared.wav', new Uint8Array([3, 4]));
    const catalog = new SpeechFixtureCatalog(directory);

    const fixtures = await catalog.refresh();
    const audio = await catalog.readAudio('shared.wav');

    expect(fixtures).toHaveLength(2);
    expect(fixtures.map(fixture => fixture.id)).toEqual(['shared.mp3', 'shared.wav']);
    expect(fixtures.every(fixture => fixture.referenceText === '共同参考')).toBe(true);
    expect(audio).toMatchObject({ id: 'shared.wav', mimeType: 'audio/wav' });
    expect([...audio.data]).toEqual([3, 4]);
  });

  it('保留非法 JSON 与缺失字段的配对项目并显式标错', async () => {
    const directory = await createFixtureDirectory();
    await addFile(directory, 'invalid.json', '{broken');
    await addFile(directory, 'invalid.wav', new Uint8Array([1]));
    await addFile(directory, 'empty.json', '{"title":"","content":""}');
    await addFile(directory, 'empty.mp3', new Uint8Array([2]));
    const catalog = new SpeechFixtureCatalog(directory);

    const fixtures = await catalog.refresh();

    expect(fixtures).toHaveLength(2);
    expect(fixtures.every(fixture => typeof fixture.error === 'string')).toBe(true);
    expect(() => catalog.resolveRunnable(['invalid.wav'])).toThrow('Fixture invalid.wav 无效');
  });

  it('目录不存在或为空时返回空列表', async () => {
    const directory = await createFixtureDirectory();
    await expect(new SpeechFixtureCatalog(directory).refresh()).resolves.toEqual([]);
    await expect(new SpeechFixtureCatalog(path.join(directory, 'missing')).refresh()).resolves.toEqual([]);
  });
});
