import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vue from '@vitejs/plugin-vue';
import { defineConfig } from 'vitest/config';

const desktopRoot = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(desktopRoot, 'renderer/src'),
    },
  },
  test: {
    environment: 'happy-dom',
    include: ['*.test.ts', 'main/**/*.test.ts', 'renderer/src/**/*.test.ts'],
  },
});
