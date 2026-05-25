import type { Span } from '../types/span';

/**
 * BPE → character offset alignment.
 *
 * Transformers.js v4 token-classification pipeline does not populate `start`/`end`
 * even with `aggregation_strategy: 'simple'` (see source: `// TODO: Add support for
 * start and end`). The `word` it returns is a `tokenizer.decode` of the grouped tokens
 * with skip_special_tokens=true, which can introduce or drop whitespace and lowercase
 * the text for uncased models. We re-align by searching the original text for the word
 * in linear order, with a whitespace-tolerant matcher.
 */

export type CharRange = { start: number; end: number };

/**
 * Try to find `word` in `text` at-or-after `fromIndex`. Returns null if no match.
 * Tolerates extra spaces inside `word` (BERT decode often inserts a space around
 * punctuation: "John , Smith" for input "John, Smith").
 */
export function findEntityCharRange(
  text: string,
  word: string,
  fromIndex: number,
): CharRange | null {
  const trimmed = word.trim();
  if (trimmed.length === 0) return null;

  // 1. Fast path: literal substring search.
  const direct = text.indexOf(trimmed, fromIndex);
  if (direct !== -1) {
    return { start: direct, end: direct + trimmed.length };
  }

  // 2. Whitespace-tolerant search. Build a regex that allows any whitespace where the
  // decoded word has whitespace, and optional whitespace around punctuation.
  const escaped = trimmed
    .split(/(\s+)/)
    .map((part) => {
      if (/^\s+$/.test(part)) return '\\s+';
      // Allow optional whitespace between adjacent characters when one side is punctuation.
      return part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    })
    .join('');
  const re = new RegExp(escaped, 'i');
  const slice = text.slice(fromIndex);
  const m = slice.match(re);
  if (m && m.index !== undefined) {
    return { start: fromIndex + m.index, end: fromIndex + m.index + m[0].length };
  }

  return null;
}

/**
 * Sanity check: every span's `start..end` must equal `span.text` substring of `text`.
 * Throws on mismatch — helps catch alignment regressions during development.
 */
export function validateOffsets(spans: Span[], text: string): void {
  for (const s of spans) {
    if (s.start < 0 || s.end > text.length || s.start >= s.end) {
      throw new Error(
        `Span out of bounds: type=${s.type} ${s.start}..${s.end}, text.length=${text.length}`,
      );
    }
    const slice = text.slice(s.start, s.end);
    if (slice !== s.text) {
      throw new Error(
        `Span text mismatch: expected ${JSON.stringify(slice)}, got ${JSON.stringify(s.text)}`,
      );
    }
  }
}
