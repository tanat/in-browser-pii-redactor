import type { Backend, FromWorker } from '../types/messages';
import type { Span, EntityType } from '../types/span';
import { extractAllRegex } from '../pipeline/regex';
import { mergeSpans } from '../pipeline/merge';
import { evaluateSpans, type EvalMetrics } from './score';

export type FixtureExpectedSpan = { type: EntityType; start: number; end: number };
export type Fixture = { id: string; text: string; expectedSpans: FixtureExpectedSpan[] };

export type FixtureResult = {
  id: string;
  predictedSpans: Span[];
  expectedSpans: FixtureExpectedSpan[];
  metrics: EvalMetrics;
  latencyMs: number;
};

export type RunReport = {
  runId: string;
  backend: Backend;
  model: string;
  perFixture: FixtureResult[];
  aggregate: EvalMetrics & { p50LatencyMs: number; p95LatencyMs: number };
  timestamp: string;
};

function makeWorker(): Worker {
  return new Worker(new URL('../worker/inference.worker.ts', import.meta.url), {
    type: 'module',
  });
}

function awaitReady(worker: Worker, forceBackend?: Backend): Promise<Backend> {
  return new Promise((resolve, reject) => {
    const onMsg = (e: MessageEvent<FromWorker>) => {
      if (e.data.type === 'ready') {
        worker.removeEventListener('message', onMsg);
        resolve(e.data.backend);
      } else if (e.data.type === 'error') {
        worker.removeEventListener('message', onMsg);
        reject(new Error(e.data.message));
      }
    };
    worker.addEventListener('message', onMsg);
    worker.postMessage({ type: 'init', forceBackend });
  });
}

function inferOnce(worker: Worker, text: string, version: number): Promise<{ spans: Span[]; latencyMs: number }> {
  return new Promise((resolve, reject) => {
    const onMsg = (e: MessageEvent<FromWorker>) => {
      if (e.data.type === 'inferred' && e.data.version === version) {
        worker.removeEventListener('message', onMsg);
        resolve({ spans: e.data.spans, latencyMs: e.data.latencyMs });
      } else if (e.data.type === 'error') {
        worker.removeEventListener('message', onMsg);
        reject(new Error(e.data.message));
      }
    };
    worker.addEventListener('message', onMsg);
    worker.postMessage({ type: 'infer', text, version });
  });
}

export type HarnessProgress = (done: number, total: number, label: string) => void;

export async function runHarness(
  fixtures: Fixture[],
  forceBackend?: Backend,
  onProgress?: HarnessProgress,
): Promise<RunReport> {
  const worker = makeWorker();
  let version = 0;
  try {
    const backend = await awaitReady(worker, forceBackend);
    const perFixture: FixtureResult[] = [];

    for (let i = 0; i < fixtures.length; i++) {
      const f = fixtures[i];
      version += 1;
      const { spans: mlSpans, latencyMs } = await inferOnce(worker, f.text, version);
      const regexSpans = extractAllRegex(f.text);
      const merged = mergeSpans(regexSpans, mlSpans);
      const metrics = evaluateSpans(merged, f.expectedSpans);
      perFixture.push({
        id: f.id,
        predictedSpans: merged,
        expectedSpans: f.expectedSpans,
        metrics,
        latencyMs,
      });
      onProgress?.(i + 1, fixtures.length, f.id);
    }

    const aggregate = aggregateReport(perFixture);
    return {
      runId: crypto.randomUUID(),
      backend,
      model: 'Xenova/bert-base-NER',
      perFixture,
      aggregate,
      timestamp: new Date().toISOString(),
    };
  } finally {
    worker.postMessage({ type: 'shutdown' });
    worker.terminate();
  }
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function aggregateReport(per: FixtureResult[]): RunReport['aggregate'] {
  // Aggregate by concatenating all predicted/expected and re-running evaluateSpans isn't
  // mathematically right since spans are per-text. Instead, sum tp/fp/fn across fixtures.
  let tp = 0;
  let fp = 0;
  let fn = 0;
  const perTypeAcc: Record<string, { tp: number; fp: number; fn: number }> = {};
  for (const f of per) {
    tp += f.metrics.truePositives;
    fp += f.metrics.falsePositives;
    fn += f.metrics.falseNegatives;
    for (const [t, m] of Object.entries(f.metrics.perType)) {
      const acc = perTypeAcc[t] ?? (perTypeAcc[t] = { tp: 0, fp: 0, fn: 0 });
      acc.tp += m.tp;
      acc.fp += m.fp;
      acc.fn += m.fn;
    }
  }
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  const sortedLatency = per.map((f) => f.latencyMs).sort((a, b) => a - b);

  const perType: EvalMetrics['perType'] = {};
  for (const [t, a] of Object.entries(perTypeAcc)) {
    const p = a.tp + a.fp === 0 ? 0 : a.tp / (a.tp + a.fp);
    const r = a.tp + a.fn === 0 ? 0 : a.tp / (a.tp + a.fn);
    const tf = p + r === 0 ? 0 : (2 * p * r) / (p + r);
    perType[t] = { precision: p, recall: r, f1: tf, ...a };
  }

  return {
    precision,
    recall,
    f1,
    truePositives: tp,
    falsePositives: fp,
    falseNegatives: fn,
    perType,
    p50LatencyMs: percentile(sortedLatency, 50),
    p95LatencyMs: percentile(sortedLatency, 95),
  };
}
