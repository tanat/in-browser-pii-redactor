import { describe, it, expect } from 'vitest';
import { spanIoU, evaluateSpans, type MinimalSpan } from '../score';

const span = (type: MinimalSpan['type'], start: number, end: number): MinimalSpan => ({
  type,
  start,
  end,
});

describe('spanIoU', () => {
  it('returns 0 when types differ', () => {
    expect(spanIoU(span('PERSON', 0, 5), span('LOCATION', 0, 5))).toBe(0);
  });
  it('returns 1 for identical spans', () => {
    expect(spanIoU(span('PERSON', 0, 5), span('PERSON', 0, 5))).toBe(1);
  });
  it('returns partial overlap', () => {
    // overlap=2, union=6 → 1/3
    expect(spanIoU(span('PERSON', 0, 4), span('PERSON', 2, 6))).toBeCloseTo(2 / 6, 5);
  });
});

describe('evaluateSpans', () => {
  it('identical sets → P=R=F1=1', () => {
    const m = evaluateSpans([span('PERSON', 0, 5)], [span('PERSON', 0, 5)]);
    expect(m.precision).toBe(1);
    expect(m.recall).toBe(1);
    expect(m.f1).toBe(1);
  });
  it('disjoint sets → P=R=F1=0', () => {
    const m = evaluateSpans([span('PERSON', 0, 5)], [span('LOCATION', 10, 20)]);
    expect(m.precision).toBe(0);
    expect(m.recall).toBe(0);
    expect(m.f1).toBe(0);
  });
  it('partial overlap above threshold → match', () => {
    // pred 0-10, exp 0-9 → IoU 9/10 = 0.9
    const m = evaluateSpans([span('PERSON', 0, 10)], [span('PERSON', 0, 9)]);
    expect(m.precision).toBe(1);
    expect(m.recall).toBe(1);
  });
  it('partial overlap below threshold → no match', () => {
    // pred 0-10, exp 7-12 → overlap 3, union 12, IoU 0.25 < 0.5
    const m = evaluateSpans([span('PERSON', 0, 10)], [span('PERSON', 7, 12)]);
    expect(m.precision).toBe(0);
    expect(m.recall).toBe(0);
  });
  it('type mismatch → no match even when offsets equal', () => {
    const m = evaluateSpans([span('PERSON', 0, 5)], [span('LOCATION', 0, 5)]);
    expect(m.precision).toBe(0);
    expect(m.recall).toBe(0);
  });
  it('per-type metrics breakdown', () => {
    const pred = [span('PERSON', 0, 5), span('LOCATION', 10, 20)];
    const exp = [span('PERSON', 0, 5)];
    const m = evaluateSpans(pred, exp);
    expect(m.perType.PERSON.f1).toBe(1);
    expect(m.perType.LOCATION.precision).toBe(0);
    expect(m.precision).toBeCloseTo(0.5, 5);
  });
});
