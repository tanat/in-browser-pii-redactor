# Web Worker — isolating ML inference from UI

## WHY

`bert-base-NER` inference on a typical phrase (~30 tokens) takes roughly this much:

| Backend | p50 latency | p95 latency |
|---------|-------------|-------------|
| WebGPU | ~45 ms | ~80 ms |
| WASM | ~200 ms | ~350 ms |

If you run this in the main thread, the main evil of browser UI happens: **event loop blocking**. On a WASM device every keystroke gives 200 ms of freeze. The cursor doesn't move, characters queue up, IME predictions break, scroll doesn't respond.

A Web Worker is a different execution context of the same origin: its own event queue, its own V8 isolate, communication only via `postMessage`. You spin it up once, load the model into it, and the main thread is unloaded. UI stays at 60 fps even on WASM-fallback devices.

Bonus: a Worker has no access to the DOM. This is useful from two sides. First, during code review it's clear: if ML is in the Worker, it can't accidentally touch `document.cookie` or anything else. Second, Transformers.js initialization (including `env.useBrowserCache = true`) doesn't pollute the page's global state.

## HOW

The contract between main thread and Worker — two union message types:

```ts
// src/types/messages.ts
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

This is a **discriminated union** in TS: both ends know the exact set of fields by `type`. If you send a Worker `{ type: 'inferr' }` — the compiler rejects it. This is the central defense against drift: the contract is described in one file, imported by both sides.

In words:

- `init` — an optional explicit init message (with the option to force the backend for benchmarks). In practice `infer` triggers init itself via `ensureInit()`, so `init` is needed only for warm-up.
- `infer` — the main working message: text + snapshot version (for stale-drop, see chapter 5).
- `progress` — model load progress events (needed for the UI progress bar on first launch).
- `ready` — signal "model loaded, I'm ready to work."
- `inferred` — result: list of `MlSpan[]`, echo of `version`, measured `latencyMs`.
- `error` — something fell over.

### The Worker itself

`src/worker/inference.worker.ts`:

```ts
/// <reference lib="webworker" />
import type { ToWorker, FromWorker } from '../types/messages';
import { initPipeline, type InitResult } from './pipeline-init';
import { setupOpfsCache } from '../storage/opfs-cache';
import { aggregateMlSpans } from './aggregate';

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;
let initState: InitResult | null = null;
let initPromise: Promise<InitResult> | null = null;
let forcedBackend: 'webgpu' | 'wasm' | undefined = undefined;

function send(msg: FromWorker) {
  ctx.postMessage(msg);
}
```

A few important details:

**`/// <reference lib="webworker" />`** — gives TS the types `DedicatedWorkerGlobalScope`, `self`, `WorkerNavigator`. Without it `self` will be typed as `Window`, and types won't line up.

**`initState` and `initPromise`** — two module-level variables implementing the "init exactly once" pattern. This is critical: loading the model is ~110 MB + compilation, it should not start twice, even if several `infer` messages arrive before it completes.

```ts
async function ensureInit(): Promise<InitResult> {
  if (initState) return initState;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    setupOpfsCache();
    const res = await initPipeline(
      (p) => {
        send({ type: 'progress', phase: p.phase, pct: p.pct, file: p.file });
      },
      forcedBackend,
    );
    initState = res;
    send({ type: 'ready', backend: res.backend, modelLoadedFrom: 'network' });
    return res;
  })();
  return initPromise;
}
```

Logic:
1. If `initState` already exists — model loaded, return immediately.
2. If `initPromise` is already running — return the same promise (a second init attempt gets the same result).
3. Otherwise create a promise that simultaneously sends progress events to the main thread.

This is the classic **double-checked promise caching**. Without it a fast message stream (init + 3 × infer) would kick off four parallel loads.

### Dispatcher

```ts
ctx.addEventListener('message', async (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'init': {
        if (msg.forceBackend) forcedBackend = msg.forceBackend;
        await ensureInit();
        return;
      }
      case 'infer': {
        const { ner } = await ensureInit();
        const t0 = performance.now();
        const result = (await ner(msg.text, { aggregation_strategy: 'simple' })) as unknown;
        const spans = aggregateMlSpans(msg.text, result);
        const latencyMs = performance.now() - t0;
        send({ type: 'inferred', spans, version: msg.version, latencyMs });
        return;
      }
      case 'shutdown': {
        ctx.close();
        return;
      }
    }
  } catch (err) {
    send({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
});
```

A few points:

**`await ensureInit()` inside `case 'infer'`** — this is lazy init: the first `infer` request will pull the model itself if it's not already there. This is convenient for cold start: the main thread can send `infer` immediately without waiting for `ready`, the answer just arrives a bit later.

**`aggregation_strategy: 'simple'`** — the mode in which Transformers.js aggregates BPE tokens into word-level entities (see chapter 4). Without it you get raw B-/I- tags on every subword, and assembling spans by hand is its own adventure.

**Measuring `latencyMs`** is done **inside the Worker**, because only here is the time "clean" — from the moment we called `ner()` to the moment we got the result. In the main thread the number would be polluted by postMessage overhead.

**A global try/catch** catches everything and sends an `error` message out. Without it the Worker could silently die from an unhandled rejection, and the main thread wouldn't understand what happened.

### Worker registration in main thread

Vite supports Workers natively via a special syntax:

```ts
// somewhere in main thread
const worker = new Worker(
  new URL('./worker/inference.worker.ts', import.meta.url),
  { type: 'module' }
);
```

A few key details:
- **`new URL(..., import.meta.url)`** — Vite intercepts this pattern at build time and bundles the worker as a separate bundle. In production the artifact will include `inference.worker-[hash].js`.
- **`type: 'module'`** — module Worker. Without this `import` inside the worker won't work. Today module workers are supported in all major browsers.

## postMessage semantics — what's worth knowing

`postMessage` uses the **structured clone algorithm**: a deep copy of the object when crossing the Worker boundary. By 2026 benchmarks, passing typical messages (`{ text: 200 chars, version: 7 }`) takes 0–1 ms, up to 80 kB/ms for large payloads. For our case this is negligible.

If the payload were large (e.g. an `ArrayBuffer` with audio), you could pass it as **transferable** — zero-copy ownership transfer: `worker.postMessage(buf, [buf])`. The difference is dramatic — on 100 MB CSV: 268 ms structured clone vs 29 ms transferable. But we have text — no point.

What to avoid: don't pass `Function`, `DOM Node`, `Error` (only the message), `Symbol`, `WeakMap/WeakSet` into `postMessage` — they don't clone. Objects with cyclic references — clone fine (structured clone supports this), but that's rarely needed.

## Worker lifecycle

A Worker lives as long as the page lives. On navigation it terminates automatically. You can terminate explicitly via `worker.terminate()` (from main thread) or `self.close()` from inside the worker — our `case 'shutdown'` does the latter.

When to terminate manually:
- On leaving the editor page (if there's SPA routing)
- On changing the model (e.g. A/B test of two different NERs)
- When the user turns off a feature flag

When not to:
- On every message. Spinning up a Worker is model compilation (~1–3 seconds on first launch, ~100 ms from cache). Terminating after every `infer` — catastrophe.

In this project the Worker is created exactly once per page lifetime. That's right for the working cycle of "editor open, user typing."

## Failure modes

**Model load fails with network error.** E.g. Hugging Face CDN is unavailable. In our pipeline's `try/catch` the `error` flies outward, the main thread shows a "couldn't load model, regex-only mode" banner. Regex-only is a working fallback — EMAIL/PHONE/SSN/DATE are still cleaned.

**WebGPU adapter failure.** On some devices `'gpu' in navigator` exists, but `requestAdapter()` returns `null` (old driver, disabled GPU). `pipeline-init.ts` handles this via try/catch + WASM fallback. More in chapter 6.

**Worker hangs.** If something inside the model went into an infinite loop (theoretically — a bug in the shader compiler), the main thread can detect this with a timer: if there's no response to `infer{version: N}` for more than 10 seconds — restart the Worker. There's no such watchdog in the current code, but it's a natural extension.

**Memory leak due to message versioning.** If main thread sends `infer` faster than Worker responds, the Worker's message queue grows. We have debounce 200 ms (see chapter 5), so the queue is practically always empty. Without debounce — potential leak.

**Browser incompatibility.** Very old browsers (< 2022) may not support module workers. In practice in 2026 this is <0.5% of traffic, can be ignored with an honest error banner.

**Dev tooling version.** The project builds on Vite 8 (Rolldown), `@vitejs/plugin-react` 6 (Oxc), Vitest 4 and TypeScript 6. Module worker import via `new URL(..., import.meta.url)` — standard Vite pattern. In `tsconfig.app.json` there's `"ignoreDeprecations": "6.0"` — TS6 marks `baseUrl` as deprecated, but we continue to keep it together with `paths`.

## Testing

The Worker is hard to test in jsdom — there's no real worker scope. We have tests for `aggregate.ts` (input — fake raw output, output — `MlSpan[]`) and `pipeline-init.ts` (mock pipeline via DI). The worker itself isn't covered by unit tests; it's covered by an e2e test via playwright (optional, not in the current code).

The main test of the Worker contract effectively is **strict message typing**. If someone changes the `ToWorker.infer` structure, TypeScript won't let them send wrong messages from the main thread. It's a "compile-time test."

## What should click

After this chapter you should understand:

1. **Why a Worker.** UI freeze on every keystroke — disqualifier. Worker eliminates the first-order problem.
2. **What a discriminated union of messages is.** One file `messages.ts`, two union types, both sides strictly typed.
3. **Why double-checked promise caching.** Without it a chain of messages launches several parallel model loads.
4. **What a module Worker is.** `type: 'module'` + `new URL(..., import.meta.url)` — the Vite-friendly way to declare one.
5. **When `transferable`, when not needed.** Text — not needed. ArrayBuffer 10+ MB — needed.

In the next chapter we'll break down what exactly the worker does at the moment of `infer`: regex pre-pass, ML inference, and merge layer — three span sources, one result.
