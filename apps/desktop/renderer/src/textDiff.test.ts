import { describe, expect, it } from 'vitest';
import { createExactTextDiff } from './textDiff.ts';

function changedText(lines: ReturnType<typeof createExactTextDiff>['hunks'][number]['referenceLines']): string {
  return lines
    .flatMap(line => line.segments)
    .filter(segment => segment.changed)
    .map(segment => segment.value)
    .join('');
}

describe('createExactTextDiff', () => {
  it('逐字保留中文替换、大小写、标点与空白差异', () => {
    const result = createExactTextDiff('你好 A，世 界', '你号 a,世界');
    const [hunk] = result.hunks;

    expect(result.identical).toBe(false);
    expect(hunk).toBeDefined();
    expect(changedText(hunk!.referenceLines)).toBe('好A， ');
    expect(changedText(hunk!.recognizedLines)).toBe('号a,');
  });

  it('过滤相同行并保留两侧真实行号', () => {
    const result = createExactTextDiff(
      '相同一\n参考二\n相同三\n参考四',
      '相同一\n识别二\n相同三\n识别四',
    );

    expect(result.hunks).toHaveLength(2);
    expect(result.hunks.map(hunk => hunk.referenceLines[0]?.lineNumber)).toEqual([2, 4]);
    expect(result.hunks.map(hunk => hunk.recognizedLines[0]?.lineNumber)).toEqual([2, 4]);
    expect(JSON.stringify(result)).not.toContain('相同一');
    expect(JSON.stringify(result)).not.toContain('相同三');
  });

  it('明确表示整行新增与整行缺失', () => {
    const inserted = createExactTextDiff('第一行\n第三行', '第一行\n第二行\n第三行');
    const removed = createExactTextDiff('第一行\n第二行\n第三行', '第一行\n第三行');

    expect(inserted.hunks[0]?.referenceLines).toEqual([{
      lineNumber: null,
      segments: [],
      missing: true,
    }]);
    expect(inserted.hunks[0]?.recognizedLines[0]?.lineNumber).toBe(2);
    expect(removed.hunks[0]?.referenceLines[0]?.lineNumber).toBe(2);
    expect(removed.hunks[0]?.recognizedLines[0]?.missing).toBe(true);
  });

  it('不忽略文件末尾换行差异', () => {
    const result = createExactTextDiff('同一行\n', '同一行');
    const [hunk] = result.hunks;

    expect(hunk?.referenceLines[0]?.segments).toContainEqual({
      value: '\n',
      changed: true,
      kind: 'line-break',
    });
    expect(hunk?.recognizedLines[0]?.lineNumber).toBe(1);
  });

  it('区分完全一致和空文本差异', () => {
    expect(createExactTextDiff('完全一致', '完全一致')).toEqual({
      identical: true,
      hunks: [],
    });

    const emptyReference = createExactTextDiff('', '识别文本');
    expect(emptyReference.identical).toBe(false);
    expect(emptyReference.hunks[0]?.referenceLines[0]?.missing).toBe(true);
    expect(changedText(emptyReference.hunks[0]!.recognizedLines)).toBe('识别文本');
  });
});
