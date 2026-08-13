import type {
  SpeechDevApi,
  SpeechDevEvent,
  SpeechRunRequest,
  SpeechRuntimePolicy,
  SpeechSynthesisRequest,
} from '../contracts.ts';

import { contextBridge, ipcRenderer } from 'electron';
import { SPEECH_DEV_CHANNELS } from '../contracts.ts';

const speechLabApi: SpeechDevApi = Object.freeze({
  getSnapshot: () => ipcRenderer.invoke(SPEECH_DEV_CHANNELS.snapshot),
  listFixtures: () => ipcRenderer.invoke(SPEECH_DEV_CHANNELS.listFixtures),
  refreshFixtures: () => ipcRenderer.invoke(SPEECH_DEV_CHANNELS.refreshFixtures),
  readFixtureAudio: (fixtureId: string) => ipcRenderer.invoke(
    SPEECH_DEV_CHANNELS.readAudio,
    fixtureId,
  ),
  readGeneratedAudio: (clipId: string) => ipcRenderer.invoke(
    SPEECH_DEV_CHANNELS.readGeneratedAudio,
    clipId,
  ),
  initialize: () => ipcRenderer.invoke(SPEECH_DEV_CHANNELS.initialize),
  setRuntimePolicy: (policy: SpeechRuntimePolicy) => ipcRenderer.invoke(
    SPEECH_DEV_CHANNELS.setRuntimePolicy,
    policy,
  ),
  stop: () => ipcRenderer.invoke(SPEECH_DEV_CHANNELS.stop),
  run: (request: SpeechRunRequest) => ipcRenderer.invoke(SPEECH_DEV_CHANNELS.run, request),
  synthesize: (request: SpeechSynthesisRequest) => ipcRenderer.invoke(
    SPEECH_DEV_CHANNELS.synthesize,
    request,
  ),
  transcribeCurrentClip: () => ipcRenderer.invoke(SPEECH_DEV_CHANNELS.transcribeCurrentClip),
  runRoundTrip: (request: SpeechSynthesisRequest) => ipcRenderer.invoke(
    SPEECH_DEV_CHANNELS.roundTrip,
    request,
  ),
  cancel: () => ipcRenderer.invoke(SPEECH_DEV_CHANNELS.cancel),
  onEvent: (listener: (event: SpeechDevEvent) => void) => {
    const wrappedListener = (_event: Electron.IpcRendererEvent, value: SpeechDevEvent): void => listener(value);
    ipcRenderer.on(SPEECH_DEV_CHANNELS.event, wrappedListener);
    return () => ipcRenderer.removeListener(SPEECH_DEV_CHANNELS.event, wrappedListener);
  },
});

contextBridge.exposeInMainWorld('speechLab', speechLabApi);
