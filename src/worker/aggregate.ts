import type { MlSpan, EntityType } from '../types/span';
import { findEntityCharRange } from '../alignment/bpe-to-char';

type RawAggregated = {
  entity_group?: string;
  score?: number;
  word?: string;
  start?: number;
  end?: number;
};

function tagToEntityType(tag: string): EntityType | null {
  switch (tag) {
    case 'PER':
      return 'PERSON';
    case 'LOC':
      return 'LOCATION';
    default:
      return null; // ORG / MISC are intentionally dropped
  }
}

/**
 * Convert Transformers.js token-classification output (with aggregation_strategy: 'simple')
 * into MlSpan[]. Recovers character offsets in the input text since the pipeline doesn't
 * populate `start`/`end` (TODO upstream — see DECISIONS.md).
 */
export function aggregateMlSpans(text: string, raw: unknown): MlSpan[] {
  if (!Array.isArray(raw)) return [];
  const spans: MlSpan[] = [];
  let cursor = 0;

  for (const r of raw as RawAggregated[]) {
    const tag = r.entity_group;
    if (!tag) continue;
    const word = r.word;
    if (!word) continue;
    const type = tagToEntityType(tag);
    if (!type) continue;

    const range = findEntityCharRange(text, word, cursor);
    if (!range) continue;
    cursor = range.end;

    spans.push({
      type,
      start: range.start,
      end: range.end,
      text: text.slice(range.start, range.end),
      source: 'ml',
      confidence: r.score ?? 0,
    });
  }
  return spans;
}
