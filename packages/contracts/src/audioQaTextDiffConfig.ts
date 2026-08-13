export type RgbHexColor = `#${string}`;

export interface AudioQaTextDiffColors {
  referenceText: RgbHexColor;
  recognizedText: RgbHexColor;
}

export const AUDIO_QA_TEXT_DIFF_COLOR_DEFAULTS = Object.freeze({
  referenceText: '#2E7D55',
  recognizedText: '#F72D5B',
} satisfies AudioQaTextDiffColors);
