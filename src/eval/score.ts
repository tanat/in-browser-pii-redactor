import type { Span, EntityType } from '../types/span';

export type MinimalSpan = { type: EntityType; start: number; end: number };

export type EvalMetrics = {
  precision: number;
  recall: number;
  f1: number;
  truePositives: number;
  falsePositives: number;
  falseNegatives: number;
  perType: Record<string, { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number }>;
};

export function spanIoU(a: MinimalSpan, b: MinimalSpan): number {
  if (a.type !== b.type) return 0;
  const overlap = Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
  const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
  return union === 0 ? 0 : overlap / union;
}

export function evaluateSpans(
  predicted: (Span | MinimalSpan)[],
  expected: (Span | MinimalSpan)[],
  iouThreshold = 0.5,
): EvalMetrics {
  const matched = new Set<number>();
  let tp = 0;
  for (const pred of predicted) {
    let bestIdx = -1;
    let bestIoU = iouThreshold;
    for (let i = 0; i < expected.length; i++) {
      if (matched.has(i)) continue;
      const iou = spanIoU(pred, expected[i]);
      if (iou >= bestIoU) {
        bestIoU = iou;
        bestIdx = i;
      }
    }
    if (bestIdx !== -1) {
      matched.add(bestIdx);
      tp += 1;
    }
  }
  const fp = predicted.length - tp;
  const fn = expected.length - tp;
  const precision = predicted.length === 0 ? 0 : tp / (tp + fp);
  const recall = expected.length === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  const types = new Set<string>([
    ...predicted.map((s) => s.type),
    ...expected.map((s) => s.type),
  ]);
  const perType: EvalMetrics['perType'] = {};
  for (const t of types) {
    const subPred = predicted.filter((s) => s.type === t);
    const subExp = expected.filter((s) => s.type === t);
    const m = evaluateOneType(subPred, subExp, iouThreshold);
    perType[t] = m;
  }
  return { precision, recall, f1, truePositives: tp, falsePositives: fp, falseNegatives: fn, perType };
}

function evaluateOneType(
  predicted: (Span | MinimalSpan)[],
  expected: (Span | MinimalSpan)[],
  iouThreshold: number,
) {
  const matched = new Set<number>();
  let tp = 0;
  for (const pred of predicted) {
    let bestIdx = -1;
    let bestIoU = iouThreshold;
    for (let i = 0; i < expected.length; i++) {
      if (matched.has(i)) continue;
      const iou = spanIoU(pred, expected[i]);
      if (iou >= bestIoU) {
        bestIoU = iou;
        bestIdx = i;
      }
    }
    if (bestIdx !== -1) {
      matched.add(bestIdx);
      tp += 1;
    }
  }
  const fp = predicted.length - tp;
  const fn = expected.length - tp;
  const precision = predicted.length === 0 ? 0 : tp / (tp + fp);
  const recall = expected.length === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1, tp, fp, fn };
}
