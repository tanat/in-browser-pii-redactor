# In-Browser PII Redactor

> Paste any text with personal information. Names, emails, phones, dates, and addresses are highlighted and redacted in place — within 200 ms after you stop typing. **Open DevTools → Network. It's empty.** All inference runs in the browser on WebGPU. The text never leaves the device.

![Privacy demo placeholder — drop a `docs/demo.gif` here showing paste → real-time redaction → empty network tab](docs/demo.gif)

## What this demonstrates

- **Privacy as a system property.** The marketing claim ("we don't send your data anywhere") is verifiable in real time from the user's own DevTools.
- **In-browser ML at production shape.** Transformers.js v4 + WebGPU, with WASM fallback, model cached in browser storage, inference offloaded to a Web Worker.
- **The two unglamorous engineering challenges.** BPE-token → DOM character offset alignment, and debounced versioned snapshots that drop stale results when the user types faster than the model.

## How it works

```
User types in <contenteditable>
        │
        ▼  MutationObserver
   debounce 200ms
        │
        ▼ snapshot { text, version++ }
        ├─► regex pre-pass (main thread): EMAIL, PHONE, SSN, DATE, ADDRESS
        │
        ▼ postMessage({ infer, text, version })
   Web Worker
        ├─► Xenova/bert-base-NER on WebGPU (or WASM)
        ├─► aggregation_strategy: 'simple' → entity_group + word
        └─► word → char offsets via cursor-following substring search
        │
        ▼ postMessage({ inferred, spans, version, latencyMs })
   Main thread
        ├─► version check (drop stale)
        ├─► merge regex ⨯ ML (4 overlap rules)
        └─► render: char-range → DOM Range → wrap with <span class="redaction">
```

The two engineering nuts:

- **`src/alignment/char-to-dom.ts`** — given character offsets in the plain-text view of a contenteditable subtree, produce a `Range` that hugs the highlight content even with `<br>` or nested element interleaving. 6 unit tests (`src/alignment/__tests__/char-to-dom.test.ts`).
- **`src/pipeline/snapshot.ts`** — `SnapshotManager` debounces text changes, increments a monotonic version per dispatch, and drops worker responses whose version is no longer current. 4 unit tests (`src/pipeline/__tests__/snapshot.test.ts`).

## Stack

| Layer | Choice |
|-------|--------|
| Build | Vite 8 (Rolldown bundler) + React 19 (no Next.js — pure SPA) |
| Language | TypeScript 6 strict |
| Styling | Tailwind v4.3 (Vite plugin) + `tailwind-merge` v3 (the version that targets Tailwind v4's class-precedence model) |
| ML runtime | `@huggingface/transformers@^4.2` |
| ML model | `Xenova/bert-base-NER` (~110 MB, ONNX, WebGPU) |
| Worker | Vanilla `Worker(..., { type: 'module' })` |
| Cache | Browser Cache Storage (file-system-backed in Chromium/Firefox, OPFS-equivalent for our read pattern) |
| Observability | IndexedDB → NDJSON export |
| Test | Vitest 4 + jsdom 26 |
| Deploy | Vercel static |

See `DECISIONS.md` for the three architecturally-load-bearing trade-offs (hybrid regex+ML, model size, cache backend).

## Running

```sh
pnpm install
pnpm dev      # http://localhost:5173 — main editor
              # http://localhost:5173/eval — evaluation dashboard
pnpm test     # 42 unit tests across regex, merge, alignment, score, snapshot
pnpm build    # produces dist/
```

First load downloads ~110 MB of model weights once. Subsequent loads hit the cache and reach "ready" in ~50 ms.

## Eval

`/eval` runs all fixtures (`src/fixtures/synthetic.json`) twice — once on WebGPU, once on WASM — and produces a downloadable `eval-results.json` with per-fixture, per-type, and aggregate precision/recall/F1, plus p50/p95 latency.

The committed `src/eval/results.json` is initially empty; populate it by running the dashboard and committing the downloaded report. The fixture corpus targets 30 hand-written texts; 10 are committed as a seed.

## Project layout

```
src/
├── alignment/      # char-to-dom Range mapping; bpe-to-char alignment
├── components/     # Editor (contenteditable + redaction render), EvalDashboard
├── eval/           # harness, benchmark, score (span-level F1 with IoU)
├── fixtures/       # synthetic.json gold corpus
├── pipeline/       # regex extractors, merge, debounce-and-version snapshot manager
├── storage/        # browser cache configuration, NDJSON IndexedDB log
├── types/          # Span, EntityType, Worker message contracts
└── worker/         # inference worker, pipeline init with WebGPU/WASM fallback
```

## Deploy

`vercel.json` is configured for SPA rewrites and the COOP/COEP headers needed for WebGPU/WASM threading. Deploy is a static push to Vercel — no env vars, no server.
