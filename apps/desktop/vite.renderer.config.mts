import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vue from '@vitejs/plugin-vue';
import { defineConfig } from 'vite';

const desktopRoot = fileURLToPath(new URL('.', import.meta.url));
const rendererRoot = path.resolve(desktopRoot, 'renderer');
const rendererSource = path.resolve(rendererRoot, 'src');

export default defineConfig({
  base: './',
  build: {
    emptyOutDir: true,
    outDir: path.resolve(desktopRoot, '.vite/renderer/main_window'),
  },
  plugins: [vue()],
  resolve: {
    alias: {
      '@': rendererSource,
    },
  },
  root: rendererRoot,
});
