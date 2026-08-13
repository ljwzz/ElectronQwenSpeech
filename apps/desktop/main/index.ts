import type { IpcMainInvokeEvent } from 'electron';
import type { SpeechDevEvent } from '../contracts.ts';

import path from 'node:path';
import process from 'node:process';
import { app, BrowserWindow, session as electronSession, ipcMain } from 'electron';
import { SPEECH_DEV_CHANNELS } from '../contracts.ts';
import { QwenLocalASRProvider } from './asr/qwenLocalASRProvider.ts';
import { SpeechFixtureCatalog } from './fixtureCatalog.ts';
import {
  requireFixtureId,
  requireNoArgument,
  requireRunRequest,
  requireRuntimePolicy,
  requireSynthesisRequest,
} from './ipcValidation.ts';
import { SpeechLabSession } from './speechLabSession.ts';
import { QwenLocalTTSProvider } from './tts/qwenLocalTTSProvider.ts';

let speechWindow: BrowserWindow | undefined;
let shutdownComplete = false;
let shutdownPromise: Promise<void> | undefined;

const fixtureCatalog = new SpeechFixtureCatalog();
const speechSession = new SpeechLabSession({
  asrProviderFactory: () => new QwenLocalASRProvider({
    onStderr: output => process.stderr.write(output),
  }),
  ttsProviderFactory: outputDirectory => new QwenLocalTTSProvider({
    onStderr: output => process.stderr.write(output),
    outputDirectory,
  }),
  emit: emitSpeechEvent,
});

function emitSpeechEvent(event: SpeechDevEvent): void {
  const window = speechWindow;
  if (window && !window.isDestroyed())
    window.webContents.send(SPEECH_DEV_CHANNELS.event, event);
}

function requireSpeechSender(event: IpcMainInvokeEvent): void {
  const window = speechWindow;
  if (
    !window
    || window.isDestroyed()
    || event.sender !== window.webContents
    || event.senderFrame !== event.sender.mainFrame
  ) {
    throw new Error('当前窗口无权调用语音诊断 IPC。');
  }
}

function registerIpcHandlers(): void {
  ipcMain.handle(SPEECH_DEV_CHANNELS.snapshot, (event, argument) => {
    requireSpeechSender(event);
    requireNoArgument(argument);
    return speechSession.snapshot();
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.listFixtures, (event, argument) => {
    requireSpeechSender(event);
    requireNoArgument(argument);
    return fixtureCatalog.list();
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.refreshFixtures, async (event, argument) => {
    requireSpeechSender(event);
    requireNoArgument(argument);
    return fixtureCatalog.refresh();
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.readAudio, async (event, fixtureId) => {
    requireSpeechSender(event);
    return fixtureCatalog.readAudio(requireFixtureId(fixtureId));
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.readGeneratedAudio, async (event, clipId) => {
    requireSpeechSender(event);
    return speechSession.readGeneratedAudio(requireFixtureId(clipId));
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.initialize, async (event, argument) => {
    requireSpeechSender(event);
    requireNoArgument(argument);
    return speechSession.initialize();
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.setRuntimePolicy, async (event, policy) => {
    requireSpeechSender(event);
    return speechSession.setRuntimePolicy(requireRuntimePolicy(policy));
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.stop, async (event, argument) => {
    requireSpeechSender(event);
    requireNoArgument(argument);
    return speechSession.stop();
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.cancel, async (event, argument) => {
    requireSpeechSender(event);
    requireNoArgument(argument);
    return speechSession.cancel();
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.run, async (event, value) => {
    requireSpeechSender(event);
    const request = requireRunRequest(value);
    const fixtures = fixtureCatalog.resolveRunnable(request.fixtureIds);
    return speechSession.run(fixtures, request.batchSize);
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.synthesize, async (event, value) => {
    requireSpeechSender(event);
    return speechSession.synthesize(requireSynthesisRequest(value));
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.transcribeCurrentClip, async (event, argument) => {
    requireSpeechSender(event);
    requireNoArgument(argument);
    return speechSession.transcribeCurrentClip();
  });
  ipcMain.handle(SPEECH_DEV_CHANNELS.roundTrip, async (event, value) => {
    requireSpeechSender(event);
    return speechSession.runRoundTrip(requireSynthesisRequest(value));
  });
}

function isAllowedNavigation(currentUrl: string, nextUrl: string): boolean {
  try {
    const current = new URL(currentUrl);
    const next = new URL(nextUrl);
    if (current.protocol === 'file:')
      return next.protocol === 'file:' && next.pathname === current.pathname;
    return next.origin === current.origin && next.pathname === current.pathname;
  } catch {
    return false;
  }
}

function configureWindowSecurity(window: BrowserWindow): void {
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-attach-webview', event => event.preventDefault());
  window.webContents.on('will-navigate', (event, url) => {
    const currentUrl = window.webContents.getURL();
    if (!currentUrl || !isAllowedNavigation(currentUrl, url))
      event.preventDefault();
  });
}

async function createSpeechWindow(): Promise<void> {
  const window = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1040,
    minHeight: 720,
    backgroundColor: '#f4f5f2',
    show: false,
    title: 'ElectronQwenSpeech · 语音诊断台',
    ...(process.platform === 'darwin' ? { titleBarStyle: 'hiddenInset' as const } : {}),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
      sandbox: true,
      webviewTag: false,
    },
  });
  speechWindow = window;
  configureWindowSecurity(window);
  window.once('ready-to-show', () => window.show());
  window.on('closed', () => {
    if (speechWindow === window)
      speechWindow = undefined;
    app.quit();
  });

  if (MAIN_WINDOW_VITE_DEV_SERVER_URL)
    await window.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  else
    await window.loadFile(path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`));
}

function beginShutdown(): Promise<void> {
  shutdownPromise ??= speechSession.dispose()
    .catch(error => process.stderr.write(`ElectronQwenSpeech shutdown failed: ${String(error)}\n`))
    .then(() => undefined);
  return shutdownPromise;
}

registerIpcHandlers();

app.whenReady().then(async () => {
  electronSession.defaultSession.setPermissionCheckHandler(() => false);
  electronSession.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
  await fixtureCatalog.refresh();
  await createSpeechWindow();
  void speechSession.initialize().catch(() => undefined);
}).catch((error) => {
  process.stderr.write(`ElectronQwenSpeech startup failed: ${String(error)}\n`);
  app.quit();
});

app.on('window-all-closed', () => app.quit());
app.on('before-quit', (event) => {
  if (shutdownComplete)
    return;
  event.preventDefault();
  void beginShutdown().finally(() => {
    shutdownComplete = true;
    app.quit();
  });
});
