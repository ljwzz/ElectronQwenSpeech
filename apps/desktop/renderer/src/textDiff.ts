import type { ChangeObject } from 'diff';

import { diffChars, diffLines } from 'diff';

export interface ExactTextDiffSegment {
  value: string;
  changed: boolean;
  kind: 'line-break' | 'text';
}

export interface ExactTextDiffLine {
  lineNumber: number | null;
  segments: ExactTextDiffSegment[];
  missing: boolean;
}

export interface ExactTextDiffHunk {
  id: number;
  referenceLines: ExactTextDiffLine[];
  recognizedLines: ExactTextDiffLine[];
}

export interface ExactTextDiffResult {
  identical: boolean;
  hunks: ExactTextDiffHunk[];
}

interface PendingHunk {
  referenceStartLine: number;
  recognizedStartLine: number;
  referenceText: string;
  recognizedText: string;
}

type DiffSide = 'reference' | 'recognized';

function appendTextSegment(
  line: ExactTextDiffLine,
  value: string,
  changed: boolean,
): void {
  if (!value)
    return;

  const previous = line.segments.at(-1);
  if (previous?.kind === 'text' && previous.changed === changed) {
    previous.value += value;
    return;
  }

  line.segments.push({
    value,
    changed,
    kind: 'text',
  });
}

function buildLines(
  changes: ChangeObject<string>[],
  side: DiffSide,
  sourceText: string,
  startLineNumber: number,
): ExactTextDiffLine[] {
  if (!sourceText) {
    return [{
      lineNumber: null,
      segments: [],
      missing: true,
    }];
  }

  const lines: ExactTextDiffLine[] = [];
  let nextLineNumber = startLineNumber;
  let currentLine: ExactTextDiffLine = {
    lineNumber: nextLineNumber,
    segments: [],
    missing: false,
  };

  for (const change of changes) {
    if ((side === 'reference' && change.added) || (side === 'recognized' && change.removed))
      continue;

    const changed = side === 'reference' ? change.removed : change.added;
    let remaining = change.value;
    let lineBreakIndex = remaining.indexOf('\n');

    while (lineBreakIndex !== -1) {
      appendTextSegment(currentLine, remaining.slice(0, lineBreakIndex), changed);
      if (changed) {
        currentLine.segments.push({
          value: '\n',
          changed: true,
          kind: 'line-break',
        });
      }
      lines.push(currentLine);
      nextLineNumber += 1;
      currentLine = {
        lineNumber: nextLineNumber,
        segments: [],
        missing: false,
      };
      remaining = remaining.slice(lineBreakIndex + 1);
      lineBreakIndex = remaining.indexOf('\n');
    }

    appendTextSegment(currentLine, remaining, changed);
  }

  if (!sourceText.endsWith('\n') || currentLine.segments.length > 0)
    lines.push(currentLine);

  return lines;
}

export function createExactTextDiff(
  referenceText: string,
  recognizedText: string,
): ExactTextDiffResult {
  const lineChanges = diffLines(referenceText, recognizedText, {
    ignoreNewlineAtEof: false,
    ignoreWhitespace: false,
    newlineIsToken: false,
    oneChangePerToken: true,
    stripTrailingCr: false,
  });
  const hunks: ExactTextDiffHunk[] = [];
  let referenceLineNumber = 1;
  let recognizedLineNumber = 1;
  let pendingHunk: PendingHunk | null = null;

  function flushPendingHunk(): void {
    if (!pendingHunk)
      return;

    const characterChanges = diffChars(
      pendingHunk.referenceText,
      pendingHunk.recognizedText,
      { ignoreCase: false },
    );
    hunks.push({
      id: hunks.length,
      referenceLines: buildLines(
        characterChanges,
        'reference',
        pendingHunk.referenceText,
        pendingHunk.referenceStartLine,
      ),
      recognizedLines: buildLines(
        characterChanges,
        'recognized',
        pendingHunk.recognizedText,
        pendingHunk.recognizedStartLine,
      ),
    });
    pendingHunk = null;
  }

  for (const change of lineChanges) {
    const unchanged = !change.added && !change.removed;
    if (unchanged) {
      flushPendingHunk();
      referenceLineNumber += change.count;
      recognizedLineNumber += change.count;
      continue;
    }

    pendingHunk ??= {
      referenceStartLine: referenceLineNumber,
      recognizedStartLine: recognizedLineNumber,
      referenceText: '',
      recognizedText: '',
    };

    if (change.removed) {
      pendingHunk.referenceText += change.value;
      referenceLineNumber += change.count;
    }
    if (change.added) {
      pendingHunk.recognizedText += change.value;
      recognizedLineNumber += change.count;
    }
  }

  flushPendingHunk();

  return {
    identical: hunks.length === 0,
    hunks,
  };
}
