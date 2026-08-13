import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    sourcemap: true,
  },
  ssr: {
    noExternal: [/^@electron-qwen-speech\//u],
  },
});
