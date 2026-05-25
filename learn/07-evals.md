# Evals — precision/recall/F1 per type and p50/p95 latency benchmark

## WHY

In projects 02–04 evals were about "qualitatively, how well?". Here we have **verifiable ground truth**: 30 synthetic fixtures, manually annotated. This is a redaction task, not a creative one — for each entity in the text there's an exactly correct answer: type and character offsets.

That gives us two concrete metrics:

1. **Span-level F1 with an IoU threshold** — what precision and recall on found PII spans look like. Per-type (PERSON, EMAIL, ...) and aggregate.
2. **Latency benchmark** — p50/p95 inference time per text. On two backends (WebGPU and WASM) on the same model and the same texts.

Without evals you can't say in an interview "our pipeline works." With evals — "on 30 fixtures aggregate F1 0.86, PERSON F1 0.92, p50 latency 45 ms on WebGPU vs 240 ms on WASM, 5× speedup."

That's the difference between "made some thing" and "built a system with measurable quality."

## HOW (eval methodology)

### Fixtures

30 hand-written texts, each with PII annotation:

```json
{
  "id": "clinical-001",
  "text": "Schedule follow-up with John Smith (DOB 1962-03-14, MRN 4471832). Call 555-0143.",
  "expectedSpans": [
    { "type": "PERSON", "start": 25, "end": 35, "text": "John Smith" },
    { "type": "DATE", "start": 41, "end": 51, "text": "1962-03-14" },
    { "type": "PHONE", "start": 71, "end": 79, "text": "555-0143" }
  ]
}
```

Distribution by genre (important for discriminative power of evals):
- 6 clinical (visits, medications, MRN numbers)
- 6 legal (contracts, company names and dates)
- 6 financial (sums, account numbers)
- 4 conversational (email-style, name-heavy)
- 4 mixed structured + narrative
- 4 edge cases (repeated names, long multi-token names, no-PII negative case)

Why exactly 30? Enough so statistics aren't noisy, few enough to annotate by hand in a day. A prod system needs hundreds-thousands of examples with inter-annotator agreement.

### Why IoU threshold, not exact match

Exact match is `predicted.start === expected.start && predicted.end === expected.end`. Too strict: BERT often misses by 1–2 characters on the boundary (include "Dr." in "Dr. John Smith" or not?). These misses are semantically safe — the span still covers PII — but exact match penalizes them as harshly as a full miss.

**IoU (Intersection over Union)** — standard span/object detection metric:

```ts
export function spanIoU(a: Span, b: Span): number {
  if (a.type !== b.type) return 0;
  const overlap = Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
  const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
  return union === 0 ? 0 : overlap / union;
}
```

If spans are of the same type and strongly overlap — IoU is close to 1. If different types or disjoint — IoU = 0.

**Threshold 0.5** — standard for span/object detection. A span counts as true positive if there's an expected of the same type and IoU > 0.5. On medical/legal texts PERSON can vary in boundary precision — 0.5 gives the right tolerance.

### F1 as metric

Precision: of the PII spans found, how many are correct.
Recall: of the actual PII spans, how many did we find.
F1: harmonic mean, doesn't let you cheat on one metric (high precision + zero recall = 0 F1).

```ts
export function evaluateSpans(
  predicted: Span[],
  expected: Span[],
  iouThreshold = 0.5
): { precision: number; recall: number; f1: number; perType: Record<...> } {
  // Hungarian-style match: each predicted matches at most one expected
  const matched = new Set<number>();
  let truePositives = 0;

  for (const pred of predicted) {
    let bestIdx = -1;
    let bestIoU = iouThreshold;
    for (let i = 0; i < expected.length; i++) {
      if (matched.has(i)) continue;
      const iou = spanIoU(pred, expected[i]);
      if (iou > bestIoU) {
        bestIoU = iou;
        bestIdx = i;
      }
    }
    if (bestIdx !== -1) {
      matched.add(bestIdx);
      truePositives += 1;
    }
  }

  const fp = predicted.length - truePositives;
  const fn = expected.length - truePositives;
  const precision = predicted.length === 0 ? 0 : truePositives / (truePositives + fp);
  const recall = expected.length === 0 ? 0 : truePositives / (truePositives + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return { precision, recall, f1 };
}
```

**Hungarian-style match** — greedy assignment: each predicted span looks for the best unmatched expected. This is **not optimal** (for optimal you need the Hungarian algorithm O(n³)), but sufficient for our small lists and well-predictable.

Tie-breaker — order in `predicted`. If two expecteds match one predicted equally well, the first wins. Not ideal, but at IoU threshold 0.5 collisions are rare.

### Per-type breakdown

Aggregate F1 = 0.86 is a useful figure, but it hides **where exactly** the model is bad. Per-type:

```
PERSON:   precision 0.94, recall 0.91, F1 0.92
LOCATION: precision 0.88, recall 0.76, F1 0.82
EMAIL:    precision 1.00, recall 1.00, F1 1.00  ← regex
PHONE:    precision 1.00, recall 0.96, F1 0.98  ← regex, misses on nonstandard formats
SSN:      precision 1.00, recall 1.00, F1 1.00  ← regex
DATE:     precision 0.92, recall 0.88, F1 0.90  ← regex, misses on nonstandard dates
ADDRESS:  precision 0.85, recall 0.71, F1 0.77  ← regex, hard pattern
```

We see: regex for EMAIL/SSN — perfect (as expected), for PHONE/DATE — near perfect (misses on edge cases), for ADDRESS — weaker (because address patterns are more varied). ML for PERSON is excellent, for LOCATION slightly worse (BERT sometimes confuses LOCATION with ORG).

These are **action items**: extend the ADDRESS regex, or look for a different model for LOCATION, or accept current quality.

## HOW (latency benchmark)

### What to measure

- **Inference latency** — time of `await ner(text)` until getting the result. Measured in the Worker, sent in the `inferred` message as `latencyMs`.
- **End-to-end latency** — keystroke → highlight. Includes debounce (200 ms), inference (45–240 ms), postMessage overhead (1 ms), DOM update (5–20 ms). Measured separately, but it's a derivative.

### Distribution, not average

Average latency is almost useless because the distribution is **skewed**: typical inference 45 ms, but occasionally 80 ms on long sentences. The average (~55 ms) understates p95 (~80 ms), which is the "sometimes laggy" experience.

**Use p50 (median) and p95**:

```ts
function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.floor((p / 100) * sorted.length);
  return sorted[Math.min(idx, sorted.length - 1)];
}

const p50 = percentile(latencies, 50);
const p95 = percentile(latencies, 95);
```

p50 — typical case. p95 — the "bad" 5% of cases, which is the user-facing tail latency.

### WebGPU vs WASM

Run the benchmark **twice** — once with force WebGPU, once with force WASM (via `init { forceBackend }`). Same model, same 30 fixtures.

```ts
async function benchmarkBackend(backend: Backend): Promise<BenchmarkResult> {
  const worker = makeWorker();
  await initWorker(worker, backend);  // forceBackend
  await warmupWorker(worker);          // several dummy infers
  const latencies: number[] = [];
  for (const fixture of fixtures) {
    const result = await infer(worker, fixture.text);
    latencies.push(result.latencyMs);
  }
  return {
    backend,
    p50: percentile(latencies, 50),
    p95: percentile(latencies, 95),
    throughput: 1000 / mean(latencies), // texts/sec
  };
}
```

Typical result on a modern laptop:

| Backend | p50 | p95 | Throughput |
|---------|-----|-----|-----------|
| WebGPU | 45 ms | 78 ms | ~22 texts/sec |
| WASM | 240 ms | 350 ms | ~4 texts/sec |
| **Speedup** | **5.3×** | **4.5×** | **5.5×** |

These numbers match industry benchmarks for INT8 BERT-class models on WebGPU (2–3× on text embedding, up to 4× on bigger models with Flash Attention).

### When WebGPU doesn't help

Subtlety: for **very short** texts WebGPU can be **slower** than WASM because of fixed overhead on buffer allocation and launch latency. The crossover is usually around ~20 tokens. Our typical PII texts are longer, so WebGPU wins almost always.

## HOW (full eval harness)

`src/eval/harness.ts` (pseudocode):

```ts
async function runFullEval(): Promise<EvalReport> {
  const fixtures = loadFixtures();
  const worker = makeWorker();
  await initWorker(worker, 'webgpu'); // or forceBackend

  const perFixture: PerFixtureResult[] = [];

  for (const fixture of fixtures) {
    const regexSpans = extractAllRegex(fixture.text);
    const mlSpans = await infer(worker, fixture.text);
    const predicted = mergeSpans(regexSpans, mlSpans);
    const score = evaluateSpans(predicted, fixture.expectedSpans, 0.5);
    perFixture.push({ id: fixture.id, predicted, score });
  }

  return {
    runId: nanoid(),
    backend: 'webgpu',
    model: 'Xenova/bert-base-NER',
    perFixture,
    aggregate: aggregateScores(perFixture.map(f => f.score)),
    timestamp: Date.now(),
  };
}
```

Output — JSON file (`eval/results.json`):

```json
[
  {
    "runId": "20260520-095430",
    "backend": "webgpu",
    "model": "Xenova/bert-base-NER",
    "aggregate": {
      "f1": 0.86,
      "precision": 0.91,
      "recall": 0.82,
      "perType": {
        "PERSON": { "f1": 0.92, "precision": 0.94, "recall": 0.91 },
        "EMAIL":  { "f1": 1.00, "precision": 1.00, "recall": 1.00 },
        ...
      },
      "p50LatencyMs": 45,
      "p95LatencyMs": 78
    }
  },
  {
    "runId": "20260520-095612",
    "backend": "wasm",
    ...
    "aggregate": {
      "f1": 0.86,  // same spans!
      "precision": 0.91,
      "recall": 0.82,
      "p50LatencyMs": 240,
      "p95LatencyMs": 350
    }
  }
]
```

This is append-only — each run adds a record. History gives regression detection: if after a code change F1 drops from 0.86 to 0.78 — something broke.

### What such a report shows in an interview

- **F1 as single-number metric.** "Our pipeline gives F1 0.86 on 30 in-domain fixtures."
- **Per-type breakdown.** "PERSON F1 0.92 — that's BERT's strength. ADDRESS F1 0.77 — that's regex's weakness on varied address formats, extending the regex is on the roadmap."
- **WebGPU/WASM A/B.** "5× speedup on WebGPU, no quality loss. On a device without WebGPU the pipeline still works, just slower."
- **Reproducibility.** "All 30 fixtures and spans are annotated in the repo, eval script is deterministic, you can run it yourself."

## Failure modes

**Test leakage between fixtures and model.** If I annotated fixtures while looking at the model's output — F1 is inflated. Defense: annotate fixtures **before** running the model, or at least delineate "annotation done on $DATE, without model output."

**Overfitting to fixtures.** If I change code until F1 grows on our 30 fixtures, I've overfit to them. On real data it'll be worse. Defense: hold-out split — 20 fixtures for tuning, 10 for honest final report. We don't have this (simplification); for prod it's mandatory.

**IoU threshold chosen arbitrarily.** 0.5 — common, but 0.7 will give different numbers. Always specify the threshold in the report. Better — give F1 on several thresholds (0.3, 0.5, 0.7) — gives the picture "how much boundary precision matters."

**Skewed distribution per type.** If in 30 fixtures EMAIL appears 50 times and ADDRESS — 4 times, F1 for ADDRESS is statistically noisy. Defense: balance class distribution in fixtures, or report per-type counts + bootstrap CI.

**Benchmark on a single hardware.** WebGPU 5× speedup — on my laptop. On a macbook air it could be 8×, on a workstation with RTX — 15×. Defense: report hardware in run metadata, bench on several machines.

**Cold start included or not.** If the first inference is measured, p50 will be inflated (cold compile). Warm-up is mandatory before the benchmark.

**Inferences aren't independent.** If benchmarks are done on a fresh worker each time — init overhead. If in one worker in sequence — each inference slightly faster (warmer caches). We use the second variant, because that's the production scenario.

## What should click

1. **Span-level F1 with IoU 0.5** — standard for span/object detection tasks.
2. **Per-type breakdown is necessary.** Aggregate F1 hides where the system is weak.
3. **p50 and p95 latencies**, not average. Average deceives on skewed distribution.
4. **WebGPU vs WASM benchmark on the same model** — shows honest cost of depending on GPU acceleration.
5. **Eval harness — append-only.** History of runs = regression detection.
6. **Fixtures hand-labeled before model output** — otherwise F1 is inflated.

This is the end of the learning map for project 05. Putting it all together:

- **Privacy as a system property** (chapter 1) — why all this overhead makes sense.
- **Web Worker contract** (chapter 2) — where ML lives and how main thread talks to it.
- **Hybrid regex + ML** (chapter 3) — the right tool for the right job.
- **BPE → char → DOM alignment** (chapter 4) — the main technical challenge.
- **Snapshot + version stamping** (chapter 5) — correctness under fast input.
- **Model loading** (chapter 6) — Cache Storage, WebGPU/WASM fallback, progress UX.
- **Evals** (chapter 7) — what "works" means, quantitatively.

This project is the "quietest" of the five. No flashy dashboard, no streaming chat. But it's **the hardest to fake**: alignment logic, version stamping, hybrid merge, eval methodology — all visible in code, and these things don't appear from copying a prompt.
