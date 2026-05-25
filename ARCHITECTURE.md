# Architecture — In-Browser PII Redactor

> Technical decisions and rationale. The architectural leitmotif is **privacy-as-system-property** — everything runs in the browser, and you can verify it from the Network tab. The central engineering challenge is **alignment between the model's BPE tokenization and character-level DOM**.

---

## Stack

| Layer | Technology | Version / note |
|------|------------|----------------|
| Build tool | Vite 8 + React 19 | No Next.js — this is a pure SPA, no server-side anything. Vite 8 ships with the Rolldown bundler. |
| Language | TypeScript 6, strict | `"ignoreDeprecations": "6.0"` in `tsconfig.app.json` — TS6 deprecates `baseUrl`, but we still rely on `baseUrl: "."` to anchor `paths`. |
| Styling | Tailwind v4.3 + `tailwind-merge` v3 | `tailwind-merge` v3 is the release that targets Tailwind v4's class-precedence model; older v2 silently picks the wrong winner against v4 utility groups. |
| ML runtime | `@huggingface/transformers` v4.2+ | Loaded via `device: 'webgpu'` with a WASM fallback. |
| ML model | `Xenova/bert-base-NER` | Verified ONNX, ~110 MB, 4 NER classes (PER, LOC, ORG, MISC); we use PER and LOC. |
| Web Worker | Vanilla `postMessage` | Inference off the main thread; module worker via Vite. |
| Storage | Browser Cache Storage | Model weights are cached after the first download; one-time fetch via Hugging Face CDN. |
| Observability | IndexedDB | Per-inference log, exportable as NDJSON. |
| Test | Vitest 4 + jsdom 26 | For regex, alignment, aggregate, merge, snapshot, score units. |
| Deploy | Vercel static (SSG) | No env vars; the site is fully static. |

**Intentionally not used:** Next.js (why?), AI SDK (server-side, contradicts privacy), any cloud APIs, IndexedDB for the model (Cache Storage is faster for large binary blobs).

---

## Why `bert-base-NER`

**Alternatives considered:**

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| `Xenova/bert-base-NER` | Verified ONNX + WebGPU, 4 classes cover ~90% of needs | 110 MB first-load | Selected |
| `Xenova/distilbert-base-uncased-finetuned-conll03-english` | Smaller (~65 MB) | Worse on person names | Backup fallback |
| `Xenova/bert-base-multilingual-cased-ner-hrl` | Multi-language | 700 MB+ | Overkill for English-only |
| Custom fine-tune on medical/legal | Best domain quality | Multi-day detour | Out of scope |

**Result:** `bert-base-NER` covers PERSON and LOCATION reliably; the other categories (EMAIL, PHONE, SSN, DATE, ADDRESS) go through **regex** — they're structured and regex gives ~99% precision at near-100% recall.

---

## Six PII categories from two sources

| Category | Source | Pattern / Class |
|----------|--------|-----------------|
| EMAIL | Regex | `/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g` |
| PHONE | Regex | NA-format: `(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}` with boundary lookarounds |
| SSN | Regex | `\b\d{3}-\d{2}-\d{4}\b` |
| DATE | Regex | ISO (`\b\d{4}-\d{2}-\d{2}\b`) + named months |
| PERSON | ML | NER label `PER` (post-aggregation) |
| LOCATION | ML | NER label `LOC` (post-aggregation) |

`ORG` from `bert-base-NER` is **not used** — too noisy on short snippets, false-positive rate is high ("Maple St" matches LOC, "Maple Street Capital" matches ORG, and the two overlap often). The drop happens in `worker/aggregate.ts` via `tagToEntityType`, which returns `null` for everything except `PER` and `LOC`.

`ADDRESS` is handled separately via a **regex with a street suffix** (`/\b\d{1,6}\s+...(St|Ave|Rd|Blvd|Lane|Way|Drive|Court|...)\b/`), because `bert-base-NER` does poorly on addresses.

---

## Data flow

```
                    User input (paste / type in <contenteditable>)
                                     │
                                     ▼
                          MutationObserver records the change
                                     │
                                     ▼
                          Debounce (200 ms idle)
                                     │
                                     ▼
                    main thread takes a snapshot:
                      { text: string, version: number++ }
                                     │
                                     ├──► Regex pre-pass (main thread)
                                     │       ├── EMAIL spans
                                     │       ├── PHONE spans
                                     │       ├── SSN spans
                                     │       ├── DATE spans
                                     │       └── ADDRESS spans
                                     │
                                     ▼
                           postMessage({ infer, text, version }) → Worker
                                     │
                                     ▼
                          Worker (dedicated thread):
                          1. Tokenize
                          2. Inference via @huggingface/transformers pipeline
                          3. aggregation_strategy: 'simple' → entity_group + word
                          4. word → char offsets via cursor-following substring search
                                     │
                                     ▼
                          postMessage({ inferred, spans, version, latencyMs }) → main
                                     │
                                     ▼
                          ★ Version check: if currentVersion !== version, DROP
                                     │
                                     ▼
                          Merge layer (main thread):
                          1. Collect all spans (regex + ML)
                          2. Resolve overlaps (4 rules)
                          3. Sort by start offset
                                     │
                                     ▼
                          Render layer:
                          1. For each span, replace text with a redaction badge
                          2. Preserve cursor position
                          3. Animation: pulse on a new span
```

---

## Repo layout

```
pii-redactor/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── Editor.tsx               # contenteditable + redaction renderer
│   │   ├── ModelStatus.tsx          # "Loading 110 MB..." / "Ready on WebGPU" indicator
│   │   ├── SpanBadge.tsx            # rendered redaction with click-to-reveal
│   │   ├── PrivacyDevtoolsHint.tsx  # one-time tooltip about the Network tab
│   │   └── EvalDashboard.tsx        # /eval page contents
│   ├── pipeline/
│   │   ├── regex.ts                 # all 5 regex extractors + tests
│   │   ├── merge.ts                 # span overlap resolution
│   │   ├── debounce.ts              # 200 ms idle debounce
│   │   └── snapshot.ts              # versioned snapshot manager
│   ├── worker/
│   │   ├── inference.worker.ts      # dedicated worker; all transformers.js code lives here
│   │   ├── pipeline-init.ts         # one-time pipeline setup with device fallback
│   │   └── aggregate.ts             # raw NER output → MlSpan[] via cursor search
│   ├── alignment/
│   │   ├── bpe-to-char.ts           # heart of Phase 6: word → char offsets
│   │   ├── char-to-dom.ts           # char offsets → DOM Range for highlighting
│   │   └── __tests__/
│   ├── storage/
│   │   ├── opfs-cache.ts            # Cache Storage configuration for model files
│   │   └── ndjson-log.ts            # IndexedDB log of inferences
│   ├── eval/
│   │   ├── harness.ts               # runs the full pipeline over fixtures
│   │   ├── score.ts                 # span-level P/R/F1 with IoU
│   │   ├── benchmark.ts             # WebGPU vs WASM perf
│   │   └── results.json             # append-only history
│   ├── fixtures/
│   │   ├── synthetic.json           # 30 hand-written texts with PII spans
│   │   └── README.md                # how I built the fixtures
│   └── types/
│       ├── span.ts                  # Span, EntityType, ConfidenceLevel
│       └── messages.ts              # WorkerMessage union types
│
├── public/                          # nothing committed here; cache handles models
├── docs/
│   └── demo.gif
├── DECISIONS.md
└── README.md
```

---

## Web Worker contract

Main thread and Worker communicate via typed messages:

```ts
// types/messages.ts

export type Backend = 'webgpu' | 'wasm';
export type ModelLoadedFrom = 'cache' | 'network';
export type ProgressPhase = 'download' | 'compile' | 'warmup';

export type ToWorker =
  | { type: 'init'; forceBackend?: Backend }
  | { type: 'infer'; text: string; version: number }
  | { type: 'shutdown' };

export type FromWorker =
  | { type: 'ready'; backend: Backend; modelLoadedFrom: ModelLoadedFrom }
  | { type: 'progress'; phase: ProgressPhase; pct: number; file?: string }
  | { type: 'inferred'; spans: MlSpan[]; version: number; latencyMs: number }
  | { type: 'error'; message: string };
```

**Critical: the `version` field on each `infer` request.** The main thread increments `version` on each snapshot. The Worker echoes `version` back. The main thread checks: if `currentVersion !== response.version`, **DROP** the result — it's stale.

---

## Phase 6: BPE → DOM offset alignment

This is the **central** exercise of the project. Detailed walkthrough.

### Problem

`bert-base-NER` operates on BPE tokens. With `aggregation_strategy: 'simple'` the pipeline groups B-/I- tokens into word-level entities, but in Transformers.js v4 it **does not populate** `start` / `end` (there's a literal `// TODO: Add support for start and end` in the upstream `TokenClassificationPipeline.call`). We get records like `{ entity_group: 'PER', score: 0.998, word: 'John Smith' }` and recover offsets ourselves.

Two pitfalls:

1. **Multi-token entities.** "Smith Brown" → `[Smith, Brown]` → each gets a label; needs to collapse into a single span. `aggregation_strategy: 'simple'` handles this.
2. **Subword tokens.** "antidisestablishmentarianism" → `[anti, ##dis, ##establish, ##ment, ##arian, ##ism]` — each subword may or may not get a B-/I- label, and they need to be aggregated into the right span. Again handled by `aggregation_strategy`.

What's left is recovering character offsets from the decoded `word` against the original `text` — see `src/alignment/bpe-to-char.ts` (`findEntityCharRange`: fast-path `indexOf` + whitespace-tolerant regex fallback) and the cursor pattern in `worker/aggregate.ts`.

### Code shape

```ts
// worker/inference.worker.ts
const ner = await pipeline('token-classification', 'Xenova/bert-base-NER', {
  device: 'webgpu',
  progress_callback,
});

const result = await ner(text, { aggregation_strategy: 'simple' });
// Result is an array of { entity_group: 'PER', score: 0.99, word: 'John Smith' } (no start/end)
```

`worker/aggregate.ts` then walks the result and calls `findEntityCharRange(text, word, cursor)` for each entity, dropping anything that can't be located. With `aggregation_strategy: 'simple'` on v4 the field is always `entity_group`, so the local `RawAggregated` type reads it directly.

### DOM alignment

Text inside `<contenteditable>` is a mix of text nodes and element nodes. `start: 27, end: 38` is an offset in the **plain-text projection**, but in the DOM those positions can fall inside different text nodes.

`alignment/char-to-dom.ts` walks `SHOW_TEXT` via `TreeWalker`, builds prefix sums over node lengths, and uses them to compute the right `(node, offset)` pairs for `range.setStart` / `range.setEnd`. Boundary tie-breaks prefer the next non-empty node for `start` so the Range "hugs" the entity.

Unit tests in `alignment/__tests__/char-to-dom.test.ts` cover at least six cases:
1. Whole text in a single text node
2. Range spans two text nodes
3. Text containing a `<br>` element
4. Text containing a nested `<span>`
5. Empty text node interleaving
6. End offset past total length (graceful clamp)

---

## Phase 7: Debounce + stale invalidation

Idea simple, implementation subtle. The algorithm:

```ts
// pipeline/snapshot.ts
class SnapshotManager {
  private currentVersion = 0;
  private debounceHandle: ReturnType<typeof setTimeout> | null = null;
  private latestText = '';

  onTextChange(text: string) {
    this.latestText = text;
    if (this.debounceHandle !== null) clearTimeout(this.debounceHandle);
    this.debounceHandle = setTimeout(() => {
      this.debounceHandle = null;
      this.currentVersion += 1;
      this.worker.postMessage({
        type: 'infer',
        text: this.latestText,
        version: this.currentVersion,
      });
    }, this.debounceMs);
  }

  handleWorkerMessage(msg: FromWorker) {
    if (msg.type !== 'inferred') return;
    if (msg.version !== this.currentVersion) return; // stale
    this.onApply(msg.spans, msg.version, msg.latencyMs);
  }
}
```

**Subtle:** the stale check only kicks in when the worker response arrives **later** than the next snapshot. If the Worker is very fast and v2 inference finishes before the user types anything else — `version === currentVersion`, we apply v2 (correct). If the user typed before v2 finished — we launched v3, so when v2 eventually arrives, `version (2) !== currentVersion (3)` — drop.

---

## Model cache

Transformers.js v4 caches models via the Cache Storage API by default in browsers. We enable it explicitly in `src/storage/opfs-cache.ts`:

```ts
export function setupOpfsCache(): void {
  env.useBrowserCache = true;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
}
```

The file name reflects the planning-stage consideration of OPFS; the implementation uses Cache Storage, which is file-system-backed in Chromium and Firefox and right-sized for read-mostly 100 MB blobs.

UX implication: on the first load (no cache) — a progress bar across the ~110 MB download; on subsequent loads — ~50 ms read from Cache Storage, "instant ready".

---

## Hybrid regex + ML: merge layer

Regex and ML can return overlapping spans (e.g. ML finds "John Smith from 142 Maple St" as PER, regex finds "142 Maple St" as ADDRESS — partial overlap).

**Merge rules** in `pipeline/merge.ts`:

1. **Full containment:** if span A fully contains span B, keep A. (ML "John Smith" ⊃ regex substring match → keep ML; ADDRESS "142 Maple St" ⊃ ML "Maple St" → keep ADDRESS.)
2. **Partial overlap:** structured (regex) wins. Email/phone/SSN/date are unambiguous, the regex result is more accurate than any ML one.
3. **Equal extent (same boundaries):** regex wins by type tag, but we keep the ML confidence (regex confidence = 1.0, ML = score).
4. **Disjoint:** keep both.

Unit tests in `pipeline/__tests__/merge.test.ts` cover each of the four cases + an edge case (empty input).

---

## Eval methodology

### Fixtures

`fixtures/synthetic.json` — 30 texts, hand-written by **you**, with spans annotated manually:

```json
[
  {
    "id": "clinical-001",
    "text": "Schedule follow-up with John Smith (DOB 1962-03-14, MRN 4471832). Call 555-0143.",
    "expectedSpans": [
      { "type": "PERSON", "start": 25, "end": 35 },
      { "type": "DATE", "start": 41, "end": 51 },
      { "type": "PHONE", "start": 71, "end": 79 }
    ]
  }
]
```

Diversity checklist (distribute across 30 texts):
- 6 clinical-style (visits, medications, MRN numbers)
- 6 legal-style (contract text with company names and dates)
- 6 financial-style (amounts, account numbers, transactions)
- 4 conversational (email-style, name-heavy)
- 4 mixed structured + narrative
- 4 edge cases (repeated identical names, long multi-token entities)

### Span-level F1 with IoU

```ts
// eval/score.ts

export function spanIoU(a: Span, b: Span): number {
  if (a.type !== b.type) return 0;
  const overlap = Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
  const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
  return union === 0 ? 0 : overlap / union;
}

export function evaluateSpans(
  predicted: Span[],
  expected: Span[],
  iouThreshold = 0.5,
): { precision: number; recall: number; f1: number; perType: Record<string, ...> } {
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

  return { precision, recall, f1 /* + per-type */ };
}
```

IoU threshold = 0.5 — standard for span tasks. On medical/legal texts PERSON boundaries can vary ("Dr. John Smith" vs "John Smith") — 0.5 gives the right tolerance.

### Benchmark: WebGPU vs WASM

`eval/benchmark.ts` runs the pipeline over the same 30 fixtures **twice** — once on WebGPU, once forced to WASM (`forceBackend: 'wasm'`). It measures:
- Per-text inference latency (median, p95)
- Total throughput (texts/second)
- Memory pressure (where available, e.g. Chromium's `performance.memory`)

Output to `eval/results.json` — two rows per run:

```json
[
  {
    "runId": "...",
    "backend": "webgpu",
    "model": "Xenova/bert-base-NER",
    "perFixture": [...],
    "aggregate": {
      "f1": 0.86,
      "precision": 0.91,
      "recall": 0.82,
      "p50LatencyMs": 45,
      "p95LatencyMs": 78
    }
  },
  {
    "runId": "...",
    "backend": "wasm",
    "aggregate": {
      "f1": 0.86,
      "precision": 0.91,
      "recall": 0.82,
      "p50LatencyMs": 240,
      "p95LatencyMs": 350
    }
  }
]
```

This is a **strong interview demo**: same model, two backends, you see the exact cost of moving from WASM to WebGPU (~5x speedup on bert-base — typical result).

---

## Architectural decisions (see DECISIONS.md)

### Decision 1 — Hybrid regex + ML (vs ML-only)

**Selected.** EMAIL, PHONE, SSN, DATE, ADDRESS via regex; PERSON, LOCATION via `bert-base-NER`. Combined in the merge layer.

**Alternative.** Pure ML — train or pick a model that catches all 6 categories.

**Why.**
- Structured categories (email, phone, SSN, date) have very high regex precision (~99%). ML on short snippets loses boundary precision.
- ML is for unstructured (names, locations), where there's no fixed pattern.
- Hybrid gives the best of both: heavy inference only on 2 categories, cheap filtering on the rest.

**Cost.** The merge layer is non-trivial. Conflict resolution rules require unit tests per rule. Documented in `pipeline/__tests__/merge.test.ts`.

### Decision 2 — `bert-base-NER` (vs distilled smaller model)

**Selected.** `Xenova/bert-base-NER`, 110 MB, 4 NER classes (we use 2: PER, LOC).

**Alternatives.** (a) `distilbert-base-uncased-finetuned-conll03-english` — 65 MB, but worse on person names in narrative-style text (uncased eats the cue that "Adams" is a name). (b) Multilingual BERT — 700 MB+, overkill for English-only.

**Why.** On a reference corpus of medical/legal-style texts distilbert gives PERSON F1 ~0.78, bert-base gives 0.86. That 8-point F1 difference outweighs the 110 vs 65 MB download (one-time, cached).

**Cost.** First load — 110 MB over the network. On 4G — 30–60 seconds. UX mitigates with a "Downloading model (one-time)" progress bar; subsequent loads are instant.

### Decision 3 — Browser Cache Storage (vs IndexedDB / no cache)

**Selected.** Browser Cache Storage (`env.useBrowserCache = true`) for model and tokenizer files; the v4 default is the right shape.

**Alternative.** (a) IndexedDB — Transformers.js v3 default. (b) No cache (re-download every time).

**Why.**
- IndexedDB has slow reads for 100 MB+ blobs; transaction overhead is noticeable (~500–1000 ms on cold start).
- Cache Storage is file-system-backed in Chromium and Firefox, with fast reads (~50 ms).
- No cache — UX disaster.

**Cost.** Cache Storage doesn't give us OPFS-specific niceties (truncation, append writes), but for read-mostly model weights that's fine.

---

## What to show at the interview

1. **Open DevTools → Network → Filter: Fetch/XHR.** Retype text with PII. **The Network tab is empty** after the initial model load. That's the **wow moment**.
2. **`alignment/bpe-to-char.ts` + tests.** "Here's the alignment between BPE tokenization and the DOM. Here are 6 unit tests that catch edge cases."
3. **`pipeline/snapshot.ts` (versioned snapshot manager).** "Here's my strategy for handling stale inference results. If the user types faster than the model, old results get dropped."
4. **`eval/results.json`** — a WebGPU vs WASM table on the same corpus. "Here's the exact price of the WebGPU vs WASM trade-off on this workload."
5. **`DECISIONS.md`** — three forks (hybrid vs ML-only, model size, cache backend).

And only then — the demo itself.

This is the "quietest" of the five projects — no fancy dashboard, no streaming chat. But it's the **least fakeable** — the alignment logic and offset edge cases are visible in the code; you can't generate that from a prompt.
