import { describe, it, expect } from 'vitest';
import { mergeSpans } from '../merge';
import type { Span } from '../../types/span';

const regex = (over: Partial<Span>): Span => ({
  type: 'EMAIL',
  start: 0,
  end: 5,
  text: 'xxxxx',
  source: 'regex',
  confidence: 1,
  ...over,
});
const ml = (over: Partial<Span>): Span => ({
  type: 'PERSON',
  start: 0,
  end: 5,
  text: 'xxxxx',
  source: 'ml',
  confidence: 0.9,
  ...over,
});

describe('mergeSpans', () => {
  it('rule 1: ML span fully contains regex span → keep ML (the larger)', () => {
    const r = regex({ type: 'ADDRESS', start: 10, end: 22 });
    const m = ml({ type: 'PERSON', start: 0, end: 30 });
    const out = mergeSpans([r], [m]);
    expect(out).toEqual([m]);
  });

  it('rule 2: partial overlap → regex wins', () => {
    const r = regex({ type: 'ADDRESS', start: 5, end: 12 });
    const m = ml({ type: 'PERSON', start: 0, end: 10 });
    const out = mergeSpans([r], [m]);
    expect(out).toEqual([r]);
  });

  it('rule 3: equal extent → regex wins on type tag', () => {
    const r = regex({ type: 'EMAIL', start: 0, end: 10 });
    const m = ml({ type: 'PERSON', start: 0, end: 10 });
    const out = mergeSpans([r], [m]);
    expect(out).toEqual([r]);
  });

  it('rule 4: disjoint spans are both kept', () => {
    const r = regex({ type: 'EMAIL', start: 0, end: 5 });
    const m = ml({ type: 'PERSON', start: 10, end: 20 });
    const out = mergeSpans([r], [m]);
    expect(out).toEqual([r, m]);
  });

  it('edge: empty inputs return empty', () => {
    expect(mergeSpans([], [])).toEqual([]);
  });
});
