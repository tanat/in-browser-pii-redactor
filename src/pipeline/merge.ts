import type { Span } from '../types/span';

/**
 * Merge regex spans (high precision, structured) with ML spans (PERSON/LOCATION) into
 * a single non-overlapping span list per the rules in ARCHITECTURE.md:
 *
 *   1. Full containment: if span A fully contains span B, keep the larger A and drop B.
 *   2. Partial overlap: structured (regex) wins over ML.
 *   3. Equal extent: regex wins on the type tag, but kept ML confidence drops.
 *      (Implementation: always prefer regex on equal extent.)
 *   4. Disjoint: keep both.
 *
 * Returns spans sorted by start ascending.
 */
export function mergeSpans(regexSpans: Span[], mlSpans: Span[]): Span[] {
  const all = [...regexSpans, ...mlSpans].sort((a, b) => a.start - b.start);
  const kept: Span[] = [];

  for (const candidate of all) {
    let drop = false;
    for (let i = 0; i < kept.length; i++) {
      const existing = kept[i];
      const rel = relate(existing, candidate);
      if (rel === 'disjoint') continue;
      // We have an interaction — resolve it.
      const winner = pickWinner(existing, candidate, rel);
      if (winner === existing) {
        drop = true;
        break;
      } else {
        // Replace the existing kept span with candidate.
        kept.splice(i, 1);
        i--; // re-check remaining kept (rare cascading overlap)
      }
    }
    if (!drop) kept.push(candidate);
  }
  return kept.sort((a, b) => a.start - b.start);
}

type Relation = 'disjoint' | 'a-contains-b' | 'b-contains-a' | 'equal' | 'partial';

function relate(a: Span, b: Span): Relation {
  if (a.end <= b.start || b.end <= a.start) return 'disjoint';
  if (a.start === b.start && a.end === b.end) return 'equal';
  if (a.start <= b.start && a.end >= b.end) return 'a-contains-b';
  if (b.start <= a.start && b.end >= a.end) return 'b-contains-a';
  return 'partial';
}

function pickWinner(a: Span, b: Span, rel: Relation): Span {
  // Equal extent: regex wins.
  if (rel === 'equal') {
    if (a.source === 'regex' && b.source !== 'regex') return a;
    if (b.source === 'regex' && a.source !== 'regex') return b;
    return a; // tie — keep first seen
  }
  // Containment: keep the larger span.
  if (rel === 'a-contains-b') return a;
  if (rel === 'b-contains-a') return b;
  // Partial overlap: regex wins.
  if (a.source === 'regex' && b.source !== 'regex') return a;
  if (b.source === 'regex' && a.source !== 'regex') return b;
  // Both regex or both ML: keep the longer one.
  return (a.end - a.start) >= (b.end - b.start) ? a : b;
}
