// @vitest-environment node

import { describe, expect, it } from 'vitest';
import { assertDevelopmentOnlyCommandAllowed } from './developmentOnly.ts';

describe('development-only Forge guard', () => {
  it.each(['package', 'make', 'publish'])('拒绝 %s 命令', (command) => {
    expect(() => assertDevelopmentOnlyCommandAllowed([
      '/usr/local/bin/node',
      '/workspace/node_modules/@electron-forge/cli/dist/electron-forge.js',
      command,
    ])).toThrow(`electron-forge ${command}`);
  });

  it('允许 start 命令', () => {
    expect(() => assertDevelopmentOnlyCommandAllowed(['electron-forge', 'start'])).not.toThrow();
  });
});
