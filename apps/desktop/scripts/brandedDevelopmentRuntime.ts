import type { IForgePlugin, StartResult } from '@electron-forge/shared-types';

import { execFile } from 'node:child_process';
import { constants, rmSync } from 'node:fs';
import { access, cp, mkdtemp, rename, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { platform, tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const PRODUCT_NAME = 'ElectronQwenSpeech';
const DEVELOPMENT_BUNDLE_ID = 'com.electronqwenspeech.desktop.dev';
const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const execFileAsync = promisify(execFile);
const requireFromDesktop = createRequire(path.join(desktopRoot, 'package.json'));

interface HelperDefinition {
  bundleName: string;
  suffix: string;
}

const helperDefinitions: readonly HelperDefinition[] = [
  { bundleName: PRODUCT_NAME, suffix: ' Helper' },
  { bundleName: `${PRODUCT_NAME} Helper (Renderer)`, suffix: ' Helper (Renderer)' },
  { bundleName: `${PRODUCT_NAME} Helper (Plugin)`, suffix: ' Helper (Plugin)' },
  { bundleName: `${PRODUCT_NAME} Helper (GPU)`, suffix: ' Helper (GPU)' },
];

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function updatePlist(plistPath: string, values: Readonly<Record<string, string>>): Promise<void> {
  for (const [key, value] of Object.entries(values)) {
    await execFileAsync('/usr/bin/plutil', [
      '-replace',
      key,
      '-string',
      value,
      plistPath,
    ]);
  }
}

interface PreparedRuntime {
  executablePath: string;
  rootPath: string;
}

async function brandHelper(frameworksPath: string, definition: HelperDefinition): Promise<void> {
  const originalName = `Electron${definition.suffix}`;
  const brandedName = `${PRODUCT_NAME}${definition.suffix}`;
  const originalAppPath = path.join(frameworksPath, `${originalName}.app`);

  if (!await pathExists(originalAppPath))
    return;

  await updatePlist(path.join(originalAppPath, 'Contents', 'Info.plist'), {
    CFBundleDisplayName: brandedName,
    CFBundleExecutable: brandedName,
    CFBundleIdentifier: `${DEVELOPMENT_BUNDLE_ID}.helper`,
    CFBundleName: definition.bundleName,
  });
  await rename(
    path.join(originalAppPath, 'Contents', 'MacOS', originalName),
    path.join(originalAppPath, 'Contents', 'MacOS', brandedName),
  );
  const brandedAppPath = path.join(frameworksPath, `${brandedName}.app`);
  await rename(originalAppPath, brandedAppPath);
}

async function prepareBrandedDevelopmentRuntime(): Promise<PreparedRuntime> {
  const electronExecutable = requireFromDesktop('electron') as unknown;
  if (typeof electronExecutable !== 'string')
    throw new TypeError('Electron executable path is unavailable.');

  const sourceAppPath = path.resolve(path.dirname(electronExecutable), '..', '..');
  const developmentRuntimeRoot = await mkdtemp(
    path.join(tmpdir(), 'electron-qwen-speech-development-runtime-'),
  );
  const developmentAppPath = path.join(developmentRuntimeRoot, `${PRODUCT_NAME}.app`);
  const contentsPath = path.join(developmentAppPath, 'Contents');
  const frameworksPath = path.join(contentsPath, 'Frameworks');

  try {
    await cp(sourceAppPath, developmentAppPath, {
      mode: constants.COPYFILE_FICLONE,
      recursive: true,
      verbatimSymlinks: true,
    });

    await updatePlist(path.join(contentsPath, 'Info.plist'), {
      CFBundleDisplayName: PRODUCT_NAME,
      CFBundleExecutable: PRODUCT_NAME,
      CFBundleIdentifier: DEVELOPMENT_BUNDLE_ID,
      CFBundleName: PRODUCT_NAME,
      LSApplicationCategoryType: 'public.app-category.productivity',
    });
    await Promise.all(
      helperDefinitions.map(definition => brandHelper(frameworksPath, definition)),
    );
    await rename(
      path.join(contentsPath, 'MacOS', 'Electron'),
      path.join(contentsPath, 'MacOS', PRODUCT_NAME),
    );

    // This is a generated development-only shell, so all nested code is re-signed ad hoc.
    await execFileAsync('/usr/bin/codesign', [
      '--force',
      '--deep',
      '--sign',
      '-',
      developmentAppPath,
    ]);
    await execFileAsync('/usr/bin/codesign', [
      '--verify',
      '--deep',
      '--strict',
      developmentAppPath,
    ]);

    return {
      executablePath: path.join(contentsPath, 'MacOS', PRODUCT_NAME),
      rootPath: developmentRuntimeRoot,
    };
  } catch (error) {
    await rm(developmentRuntimeRoot, { force: true, recursive: true });
    throw error;
  }
}

export function createBrandedDevelopmentRuntimePlugin(): IForgePlugin {
  const activeRuntimeRoots = new Set<string>();
  let pendingRuntimeRoot: string | undefined;

  return {
    __isElectronForgePlugin: true,
    getHooks: () => ({
      postStart: async (_forgeConfig, appProcess) => {
        const runtimeRoot = pendingRuntimeRoot;
        pendingRuntimeRoot = undefined;
        if (!runtimeRoot)
          return;

        appProcess.once('exit', () => {
          void rm(runtimeRoot, { force: true, recursive: true })
            .then(() => activeRuntimeRoots.delete(runtimeRoot))
            .catch(() => undefined);
        });
      },
    }),
    init: () => {
      process.once('exit', () => {
        for (const runtimeRoot of activeRuntimeRoots)
          rmSync(runtimeRoot, { force: true, recursive: true });
      });
    },
    name: 'branded-development-runtime',
    startLogic: async (): Promise<StartResult> => {
      if (platform() !== 'darwin')
        return false;

      const runtime = await prepareBrandedDevelopmentRuntime();
      activeRuntimeRoots.add(runtime.rootPath);
      pendingRuntimeRoot = runtime.rootPath;
      return runtime.executablePath;
    },
  };
}
