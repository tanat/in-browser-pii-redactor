# Learning Map — In-Browser PII Redactor

Projects 02–04 follow the same scheme: the browser collects data, the server forwards it to a cloud model, the text leaves the device perimeter. Project 05 breaks that scheme. Open DevTools → Network, type "Called Ivan Petrov, +7 999 123-45-67" — Network is empty. Not a single outgoing request. The BERT model (~110 MB, ONNX) is loaded into the browser once, cached in Cache Storage, and from then on runs fully offline on your GPU via WebGPU.

This isn't an API budget optimization. It's a different trust model: data physically cannot leak because there's no code that would send it anywhere. An open Network tab is part of the demo. When you show this to people in an interview or in a security-sensitive chat, an empty tab is more convincing than any NDA.

This stack became production-ready: Transformers.js v4 shipped with a C++-rewritten WebGPU runtime, WebGPU is on by default in all major browsers (Safari 26 with macOS Tahoe, Firefox 141 on Windows, Firefox 145 on macOS ARM64), and Cache Storage is officially recommended by the Chrome team as primary storage for AI models up to several hundred megabytes.

---

## What this project teaches

Six concrete engineering problems that didn't exist in previous projects:

**Privacy as a system property.** Not "we promise not to log," but "the network stack physically isn't used." This guarantee is verified at a glance in DevTools. Architecturally that means: no Next.js, no server-side code, no API routes. A single-page app, static deploy, everything remote — only the initial download of model weights.

**Web Workers for ML inference.** `bert-base-NER` via the WASM backend takes 150–300ms on a typical phrase. In the main thread that's a full UI freeze on every keystroke. A Worker isolates inference in a separate thread; the main thread keeps reacting to input, rendering the cursor, handling events. The contract between them — typed `ToWorker` / `FromWorker` messages.

**BPE → character → DOM alignment.** Transformers.js v4 in the `token-classification` pipeline with `aggregation_strategy: 'simple'` **does not fill** the `start`/`end` fields in the result — inside the library there's literally `// TODO: Add support for start and end`. You have to reconstruct character offsets manually from the decoded `word` field (`findEntityCharRange`), then map these char offsets onto a DOM Range via TreeWalker (`charRangeToDomRange`). This is the central algorithm of the project.

**Versioned snapshot pattern.** The user types faster than the model responds. Worker responses can arrive out of order. `SnapshotManager` assigns each request an incremental version; responses with an old version are dropped without processing. Without this pattern a fast typist will see "zombie spans" from text they've already erased.

**Hybrid regex + ML pipeline.** BERT is good at catching names and place names (unstructured), regex catches EMAIL/PHONE/SSN/DATE/ADDRESS (structured, precision ~1.0). A merge layer combines two span streams via four rules — containment, partial overlap, equal extent, disjoint. This is the "right tool for the right job" pattern instead of "one big model for everything."

**WebGPU vs WASM benchmark.** Same model, same weights, same spans. ~45ms on WebGPU vs ~240ms on WASM on a typical phrase. The benchmark is built into the eval harness, the numbers are reproducible. In an interview this shows that you understand not just "the model works" but **at what cost**.

---

## Stage map

| # | File | What's taught | Difficulty |
|---|------|---------------|-----------|
| 1 | `01-mental-model.md` | Privacy as a system property; in-browser ML stack 2026 | Low |
| 2 | `02-worker.md` | Web Worker contract; typed messages; lifecycle; `ensureInit` | Medium |
| 3 | `03-hybrid-pipeline.md` | Regex + ML; merge layer; four overlap rules | Medium |
| 4 | `04-alignment.md` | **Main exercise** — BPE → char → DOM | High |
| 5 | `05-snapshot.md` | Debounce + versioning; stale invalidation | Medium |
| 6 | `06-model-loading.md` | WebGPU / WASM fallback; Cache Storage vs OPFS | Medium |
| 7 | `07-evals.md` | Span IoU F1 per-type; p50/p95 latency; backend benchmark | Medium |

---

## Stack — not Next.js (this is principled)

The only one of five projects without Next.js. Pure Vite + React 19 SPA + TypeScript strict + Tailwind v4, no server-side. If there were a server, the model could accidentally start running there, and the privacy guarantee would fall apart. The absence of a server is an architectural requirement, not a simplification.

```
Vite 8             — dev server + bundling on Rolldown, nothing server-side
@vitejs/plugin-react 6 — Oxc instead of Babel, faster transform
React 19           — UI, contenteditable, rendering badges
TypeScript 6 strict — typing for message contract, span types
                     (tsconfig.app.json: "ignoreDeprecations": "6.0" — TS6
                      marks baseUrl as deprecated, we keep it alongside
                      paths, the flag silences the warning)
Tailwind v4.3      — styles, JIT via Vite plugin
tailwind-merge v3  — version targeting Tailwind v4 class-precedence
@huggingface/transformers@^4.2  — ML pipeline, ONNX Runtime Web under the hood
Web Worker         — isolating inference from the main thread
Cache Storage      — storing model weights (~110 MB) between sessions
Vitest 4 + jsdom 26 — unit tests for regex / merge / alignment / aggregate
```

Deploy — Vercel static. No environment variables, no secrets, nothing server-side. Each commit turns into an immutable CDN artifact; the user downloads HTML + JS + (once) the model weights — and that's it.

---

## Quick code orientation

```
src/worker/
  inference.worker.ts   ← the only place where @huggingface/transformers lives;
                          message dispatcher: init / infer / shutdown
  pipeline-init.ts      ← initPipeline() with WebGPU → WASM fallback via try/catch
  aggregate.ts          ← raw NER output → MlSpan[] via cursor pattern over bpe-to-char

src/alignment/
  bpe-to-char.ts        ← findEntityCharRange(): fast path indexOf + whitespace-tolerant regex
  char-to-dom.ts        ← charRangeToDomRange(): TreeWalker → prefix sums → Range

src/pipeline/
  regex.ts              ← EMAIL_RE / PHONE_RE / SSN_RE / DATE_*_RE / ADDRESS_RE
  merge.ts              ← mergeSpans(): relate() → pickWinner() by 4 rules
  snapshot.ts           ← SnapshotManager: debounce 200ms + version-based stale drop

src/types/
  span.ts               ← Span, EntityType, MlSpan, source: 'regex' | 'ml'
  messages.ts           ← ToWorker / FromWorker union types; Backend; ProgressPhase

src/storage/
  opfs-cache.ts         ← env.useBrowserCache=true; setup Cache Storage in Worker scope

src/eval/
  score.ts              ← spanIoU(), evaluateSpans() — span-level F1
  benchmark.ts          ← WebGPU vs WASM latency + F1 comparison
  harness.ts            ← pipeline over 30 synthetic fixtures

src/fixtures/
  synthetic.json        ← 30 texts with manually annotated expectedSpans
```

---

## Connection to previous projects

| Concept | Projects 02–04 (AI SDK + cloud) | Project 05 (in-browser) |
|-----------|---------------------------------|------------------------|
| AI SDK | Central tool | Not used |
| Model | API to the cloud (gpt-5, claude-4.7) | ONNX in the browser via `@huggingface/transformers` |
| Latency bottleneck | Network + TTFT (~200–2000 ms) | Model load once, then ~45 ms/inference |
| Privacy | "We don't log," trust in the provider | Data physically doesn't leave the browser |
| Structured output | `streamObject` / `generateObject` | `Span[]` with char offsets |
| Evals | nDCG, trajectory metrics | Span IoU F1 + WebGPU vs WASM benchmark |
| Stale invalidation | sessionId check during streaming | version check in SnapshotManager |
| Where inference lives | Server route handler | Web Worker in the browser |

---

## State of the stack

This is worth pinning down, because three years ago this project would've been exotic, a year ago an experiment, today the norm.

- **Transformers.js v4** (February 2026) — WebGPU runtime rewritten in C++, up to 4× speedup browser-side, build time down from 2s to 200ms, bundle 53% smaller. Pipeline API is v3-compatible.
- **WebGPU** — Chrome/Edge for a while now, Safari 26 (macOS Tahoe, iOS 26), Firefox 141 (Windows), Firefox 145 (macOS ARM64). Linux in Firefox — promised for 2026. Coverage sufficient for default-on with WASM fallback.
- **ONNX Runtime Web** — under the hood of Transformers.js, WebGPU EP got Flash Attention, graph capture, Split-K MatMul. INT8 BERT-class models — interactive latency on typical hardware.
- **Cache Storage vs OPFS** — for weights up to ~500 MB the Chrome team recommends **Cache Storage** (which is what Transformers.js does by default via `env.useBrowserCache = true`). OPFS is better for multi-GB files with streaming reads.
- **Alternatives to BERT-base-NER** — GLiNER small/medium v2.1 (Apache 2.0), NuNER. Today they give better zero-shot, but for production with well-tested PII categories `bert-base-NER` remains a reliable baseline with a verified ONNX build.

---

## Additional material

- [Transformers.js v4 release notes](https://github.com/huggingface/transformers.js/releases/tag/4.0.0) — what changed, new WebGPU runtime
- [Transformers.js docs](https://huggingface.co/docs/transformers.js) — pipeline API, env config, supported tasks
- [bert-base-NER on Hugging Face](https://huggingface.co/dslim/bert-base-NER) — model card: data, metrics, tags
- [MDN Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API) — lifecycle, postMessage, transferables
- [MDN WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) — browser GPU API
- [Can I Use: WebGPU](https://caniuse.com/webgpu) — current support table
- [Chrome AI: Cache models in the browser](https://developer.chrome.com/docs/ai/cache-models) — why Cache Storage, not IndexedDB
- [GLiNER](https://github.com/urchade/GLiNER) — for those who want zero-shot NER in the browser (state of the art 2026)
