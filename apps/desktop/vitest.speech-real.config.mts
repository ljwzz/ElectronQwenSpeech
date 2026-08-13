import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['main/speechLabSession.real.ts'],
    fileParallelism: false,
    testTimeout: 900_000,
    hookTimeout: 900_000,
  },
});
