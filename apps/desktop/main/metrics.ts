export function normalizeSpeechText(value: string): string {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]/gu, '');
}

export function countSpeechCharacters(value: string): number {
  return [...normalizeSpeechText(value)].length;
}

export function calculateSpeechSimilarity(actual: string, expected: string): number {
  const actualCharacters = [...normalizeSpeechText(actual)];
  const expectedCharacters = [...normalizeSpeechText(expected)];
  const maximumLength = Math.max(actualCharacters.length, expectedCharacters.length);
  if (maximumLength === 0)
    return 1;

  let previousRow = expectedCharacters.map((_, index) => index + 1);
  previousRow.unshift(0);
  for (const [actualIndex, actualCharacter] of actualCharacters.entries()) {
    const currentRow = [actualIndex + 1];
    for (const [expectedIndex, expectedCharacter] of expectedCharacters.entries()) {
      currentRow.push(Math.min(
        currentRow[expectedIndex]! + 1,
        previousRow[expectedIndex + 1]! + 1,
        previousRow[expectedIndex]! + (actualCharacter === expectedCharacter ? 0 : 1),
      ));
    }
    previousRow = currentRow;
  }
  return 1 - previousRow[expectedCharacters.length]! / maximumLength;
}

export function calculateCpm(characterCount: number, seconds: number): number | null {
  if (!Number.isFinite(seconds) || seconds <= 0)
    return null;
  return characterCount / seconds * 60;
}
