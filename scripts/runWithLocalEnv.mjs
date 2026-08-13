import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { parseEnv } from 'node:util';

export const MODEL_PATH_ENVIRONMENT_VARIABLES = Object.freeze([
  'ELECTRON_QWEN_SPEECH_TTS_MODEL_PATH',
  'ELECTRON_QWEN_SPEECH_ASR_MODEL_PATH',
  'ELECTRON_QWEN_SPEECH_ALIGNER_MODEL_PATH',
]);

const MODULE_PATH = fileURLToPath(import.meta.url);
const REPOSITORY_ROOT = path.resolve(path.dirname(MODULE_PATH), '..');
export const LOCAL_ENV_PATH = path.join(REPOSITORY_ROOT, '.env.local');

export function loadLocalModelEnvironment(
  environment = process.env,
  envPath = LOCAL_ENV_PATH,
) {
  const result = { ...environment };
  if (!existsSync(envPath))
    return result;

  const localEnvironment = parseEnv(readFileSync(envPath, 'utf8'));
  for (const variable of MODEL_PATH_ENVIRONMENT_VARIABLES) {
    if (!(variable in environment) && localEnvironment[variable] !== undefined)
      result[variable] = localEnvironment[variable];
  }
  return result;
}

export function runWithLocalEnvironment(arguments_ = process.argv.slice(2)) {
  const [command, ...commandArguments] = arguments_;
  if (!command)
    throw new Error('缺少要执行的命令。');

  const result = spawnSync(command, commandArguments, {
    env: loadLocalModelEnvironment(),
    stdio: 'inherit',
  });
  if (result.error)
    throw result.error;
  if (result.signal) {
    process.kill(process.pid, result.signal);
    return 1;
  }
  return result.status ?? 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === MODULE_PATH) {
  try {
    process.exitCode = runWithLocalEnvironment();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
