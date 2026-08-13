import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['main/tts/qwenLocalTTSProvider.real.ts'],
    fileParallelism: false,
    testTimeout: 600_000,
    hookTimeout: 600_000,
  },
});
