import type {
  SpeechRunRequest,
  SpeechRuntimePolicy,
  SpeechSynthesisRequest,
} from '../contracts.ts';

import { SPEECH_DEV_RUNTIME_CONFIG } from '../contracts.ts';

export function requireNoArgument(value: unknown): void {
  if (value !== undefined)
    throw new Error('该操作不接受参数。');
}

export function requireFixtureId(value: unknown): string {
  if (typeof value !== 'string' || !value.trim())
    throw new Error('fixtureId 必须是非空字符串。');
  return value;
}

export function requireRuntimePolicy(value: unknown): SpeechRuntimePolicy {
  if (
    typeof value !== 'string'
    || !(SPEECH_DEV_RUNTIME_CONFIG.policies as readonly string[]).includes(value)
  ) {
    throw new Error('MPS 策略必须是 resident 或 exclusive。');
  }
  return value as SpeechRuntimePolicy;
}

export function requireSynthesisRequest(value: unknown): SpeechSynthesisRequest {
  if (typeof value !== 'object' || value === null || Array.isArray(value))
    throw new Error('TTS 参数必须是对象。');
  const request = value as Record<string, unknown>;
  const unknownKeys = Object.keys(request).filter(
    key => key !== 'text' && key !== 'voiceId' && key !== 'instruction',
  );
  if (unknownKeys.length > 0)
    throw new Error(`TTS 参数包含未知字段：${unknownKeys.join(', ')}`);
  if (typeof request.text !== 'string' || !request.text.trim())
    throw new Error('text 必须是非空字符串。');
  if (typeof request.voiceId !== 'string' || !request.voiceId.trim())
    throw new Error('voiceId 必须是非空字符串。');
  if (request.instruction !== undefined && typeof request.instruction !== 'string')
    throw new Error('instruction 必须是字符串。');
  return {
    text: request.text,
    voiceId: request.voiceId,
    ...(request.instruction === undefined ? {} : { instruction: request.instruction }),
  };
}

export function requireRunRequest(value: unknown): SpeechRunRequest {
  if (typeof value !== 'object' || value === null || Array.isArray(value))
    throw new Error('运行参数必须是对象。');
  const request = value as Record<string, unknown>;
  const unknownKeys = Object.keys(request).filter(key => key !== 'fixtureIds' && key !== 'batchSize');
  if (unknownKeys.length > 0)
    throw new Error(`运行参数包含未知字段：${unknownKeys.join(', ')}`);
  if (!Array.isArray(request.fixtureIds) || request.fixtureIds.length === 0)
    throw new Error('fixtureIds 必须是非空数组。');
  if (!request.fixtureIds.every(fixtureId => typeof fixtureId === 'string' && fixtureId.trim()))
    throw new Error('fixtureIds 必须全部为非空字符串。');
  const fixtureIds = request.fixtureIds as string[];
  if (new Set(fixtureIds).size !== fixtureIds.length)
    throw new Error('fixtureIds 不能重复。');
  if (!Number.isSafeInteger(request.batchSize) || (request.batchSize as number) <= 0)
    throw new Error('batchSize 必须是正安全整数。');
  return { fixtureIds: [...fixtureIds], batchSize: request.batchSize as number };
}
