import type { ForgeConfig } from '@electron-forge/shared-types';

import process from 'node:process';
import { VitePlugin } from '@electron-forge/plugin-vite';
import { assertDevelopmentOnlyCommandAllowed } from './developmentOnly.ts';
import { createBrandedDevelopmentRuntimePlugin } from './scripts/brandedDevelopmentRuntime.ts';

assertDevelopmentOnlyCommandAllowed(process.argv);

const config: ForgeConfig = {
  hooks: {
    prePackage: async () => {
      assertDevelopmentOnlyCommandAllowed(['package']);
    },
  },
  packagerConfig: {
    appBundleId: 'com.electronqwenspeech.desktop',
    appCategoryType: 'public.app-category.productivity',
    asar: true,
    executableName: 'ElectronQwenSpeech',
    name: 'ElectronQwenSpeech',
  },
  makers: [],
  plugins: [
    createBrandedDevelopmentRuntimePlugin(),
    new VitePlugin({
      build: [
        {
          entry: { main: 'main/index.ts' },
          config: 'vite.main.config.mts',
        },
        {
          entry: { preload: 'preload/index.ts' },
          config: 'vite.preload.config.mts',
          target: 'preload',
        },
      ],
      concurrent: false,
      renderer: [
        {
          name: 'main_window',
          config: 'vite.renderer.config.mts',
        },
      ],
    }),
  ],
};

export default config;
