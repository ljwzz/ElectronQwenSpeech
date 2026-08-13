import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, test } from 'node:test';
import {
  loadLocalModelEnvironment,
  MODEL_PATH_ENVIRONMENT_VARIABLES,
} from './runWithLocalEnv.mjs';

const temporaryDirectories = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0))
    rmSync(directory, { force: true, recursive: true });
});

function createEnvFile(contents) {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'electron-qwen-speech-env-test-'));
  const envPath = path.join(directory, '.env.local');
  temporaryDirectories.push(directory);
  writeFileSync(envPath, contents, 'utf8');
  return envPath;
}

test('加载 .env.local 中的三项模型路径并忽略其它变量', () => {
  const envPath = createEnvFile(`
ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH="/models/tts path"
ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH=/models/asr
ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH=/models/aligner
UNRELATED_SECRET=not-forwarded
`);

  const environment = loadLocalModelEnvironment({}, envPath);

  assert.deepEqual(environment, {
    ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH: '/models/aligner',
    ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH: '/models/asr',
    ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH: '/models/tts path',
  });
});

test('父进程环境变量优先于 .env.local', () => {
  const envPath = createEnvFile(MODEL_PATH_ENVIRONMENT_VARIABLES
    .map(variable => `${variable}=/models/from-file`)
    .join('\n'));
  const environment = loadLocalModelEnvironment({
    ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH: '/models/from-shell',
  }, envPath);

  assert.equal(
    environment.ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH,
    '/models/from-shell',
  );
  assert.equal(
    environment.ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH,
    '/models/from-file',
  );
});

test('.env.local 不存在时保留父进程环境', () => {
  assert.deepEqual(
    loadLocalModelEnvironment({ EXISTING: 'value' }, '/missing/.env.local'),
    { EXISTING: 'value' },
  );
});
