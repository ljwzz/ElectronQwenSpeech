import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

export function findDevelopmentRepositoryRoot(startDirectory = process.cwd()): string {
  let currentDirectory = path.resolve(startDirectory);
  while (true) {
    if (
      existsSync(path.join(currentDirectory, 'pnpm-workspace.yaml'))
      && existsSync(path.join(currentDirectory, 'services', 'asr-sidecar', 'main.py'))
    ) {
      return currentDirectory;
    }
    const parentDirectory = path.dirname(currentDirectory);
    if (parentDirectory === currentDirectory)
      throw new Error(`无法从当前目录定位 ElectronQwenSpeech 开发仓库：${startDirectory}`);
    currentDirectory = parentDirectory;
  }
}
