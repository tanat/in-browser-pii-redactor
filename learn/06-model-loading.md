# Model loading — WebGPU/WASM fallback and Cache Storage

## WHY

This is the slowest moment of the pipeline and the most visible phase for the user.

First launch: download `model.onnx` (~110 MB), tokenizer JSON, config. Compile the ONNX graph for the backend (WebGPU shader compilation or WASM kernel selection). Warm up — run an empty input through the graph so the first "real" inference is fast. Total time — 10–30 seconds on 4G, seconds on fast wifi.

Each subsequent launch: read from Cache Storage (~50 ms for a 110 MB blob), re-create the pipeline (~200–500 ms), warm up. Total ~1 second.

Without proper storage and fallback this all falls apart. Downloading 110 MB on every visit is a UX catastrophe. Crashing on a device without WebGPU means losing half the users (Firefox on Linux is still without WebGPU, old macOS — too). Not showing progress — the user will decide the page is broken.

This chapter is about three coordinated decisions:

1. **Backend selection with try/catch fallback** — we prefer WebGPU, falls back to WASM on error.
2. **Cache Storage** for model weights — the Chrome team recommends precisely it, and Transformers.js uses it via `env.useBrowserCache`.
3. **Progress events** — the main thread sees the load progressing and shows a progress bar.

## HOW (backend selection)

`src/worker/pipeline-init.ts`:

```ts
import { pipeline, type PipelineType, type PretrainedModelOptions } from '@huggingface/transformers';
import type { Backend, ProgressPhase } from '../types/messages';

export const MODEL_ID = 'Xenova/bert-base-NER';
export const TASK: PipelineType = 'token-classification';

export type ProgressEvent = { phase: ProgressPhase; pct: number; file?: string };
export type ProgressFn = (e: ProgressEvent) => void;

export type InitResult = {
  ner: (text: string, opts?: Record<string, unknown>) => Promise<unknown>;
  backend: Backend;
};

export function selectDevice(): Backend {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    return 'webgpu';
  }
  return 'wasm';
}
```

Note: we import `PretrainedModelOptions` directly from `@huggingface/transformers` and type `opts` with it — honest typing, without any-casts.

### `selectDevice` — capability detection

`'gpu' in navigator` is a feature detect for WebGPU. If the `navigator.gpu` object exists — the browser promises support. This is **necessary, but not sufficient**: the user may have an outdated driver, and `navigator.gpu.requestAdapter()` will return `null` (which Transformers.js will detect when creating the pipeline and throw).

So the full solution — **try WebGPU, fall back to WASM on error**:

```ts
export async function initPipeline(
  onProgress: ProgressFn,
  forceBackend?: Backend,
): Promise<InitResult> {
  const tryDevice = async (device: Backend): Promise<InitResult> => {
    const opts: PretrainedModelOptions = {
      device,
      progress_callback: (p: unknown) => adaptProgress(p as HfProgress, onProgress),
    };
    const ner = (await pipeline(TASK, MODEL_ID, opts)) as unknown as InitResult['ner'];
    return { ner, backend: device };
  };

  if (forceBackend) {
    return await tryDevice(forceBackend);
  }
  const preferred = selectDevice();
  if (preferred === 'webgpu') {
    try {
      return await tryDevice('webgpu');
    } catch (err) {
      console.warn('[pipeline-init] WebGPU failed, falling back to WASM:', err);
      return await tryDevice('wasm');
    }
  }
  return await tryDevice('wasm');
}
```

Logic:

1. **`forceBackend`** (optional, for benchmarks) — try exactly that, no fallback. Useful in `eval/benchmark.ts`, where we want to compare honestly: the same model on two backends.
2. **`selectDevice()` says WebGPU** — try it, if it falls (e.g. adapter == null, or GPU OOM during shader compilation) — fall back to WASM. Log a warning, don't throw.
3. **`selectDevice()` says WASM** — go straight to WASM, no try/catch (WASM doesn't fall in normal conditions, and in the worst case Transformers.js itself will throw, which we want to see).

### When the backend honestly fails

A few real scenarios:

- **Old driver.** `navigator.gpu` exists, `requestAdapter` returns `null`. Throws inside `pipeline()`.
- **GPU OOM at shader compile.** Very old low-end GPU (e.g. Intel HD Graphics older than 5th gen). Shader compilation may fall.
- **Browser flags.** Someone disabled WebGPU via chrome://flags. `'gpu' in navigator` === false. `selectDevice()` immediately gives WASM.
- **Headless Chromium without GPU.** Old `--headless` mode. `selectDevice()` → WASM.

In each case we land on WASM with known latency (~200 ms p50). This is **honest degradation**: the model works, just slower.

## HOW (Cache Storage)

`src/storage/opfs-cache.ts`:

```ts
import { env } from '@huggingface/transformers';

/**
 * Configure Transformers.js to cache models in browser storage.
 *
 * Transformers.js v4 caches models via the Cache Storage API by default in browsers.
 * OPFS isn't a separately selectable backend in v4 — the library prefers `caches`
 * (Cache Storage), which is file-system-backed in Chromium and behaves similarly to
 * OPFS for read latency.
 */
export function setupOpfsCache(): void {
  // TODO: migrate to ModelRegistry-based config in a future Transformers.js refactor.
  env.useBrowserCache = true;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
}
```

Despite the filename `opfs-cache.ts` (historical, we considered OPFS at the planning stage), in fact this is Cache Storage. There are fundamental reasons for this decision, let's go through them.

### Cache Storage vs OPFS vs IndexedDB — practical consensus

The Chrome team's recommendation is the following (see https://developer.chrome.com/docs/ai/cache-models):

| Storage | When to use | Performance |
|---------|-------------------|-------------------|
| **Cache Storage** | Default for AI models up to ~500 MB | Read: streaming-friendly, ~50 ms for 100 MB blob (Chrome) |
| **OPFS** | For multi-GB models with streaming reads | Real filesystem; ~90 ms per 100 MB write |
| IndexedDB | Don't use for binary blobs | ~850 ms per 100 MB write — slow due to transaction overhead |

Main advantages of Cache Storage for our case:

- **Match by URL** — Transformers.js does `fetch('https://huggingface.co/.../model.onnx')`, and the cache intercepts that URL naturally. No need for custom logic "turn a URL into a filename, look it up in OPFS."
- **Streaming friendly** — the model is read via `Response.body` (ReadableStream), no need to load it into RAM in full.
- **Standard API** — `caches.open` + `cache.match` + `cache.put`. OPFS requires FileSystemSyncAccessHandle, which is available only in Workers and has different semantics.

OPFS becomes preferable when:
- Weights >500 MB, don't fit in a single HTTP response without issues.
- Random access is needed (read only part of the weights).
- You want a sync API from the Worker (FileSystemSyncAccessHandle).

Our 110 MB — Cache Storage is a clean win.

### What the flags actually do

- **`env.useBrowserCache = true`** — Transformers.js intercepts its fetches through Cache Storage. The first request — hits network, response is cached. Subsequent — hit cache, hit network only on cache miss.
- **`env.allowRemoteModels = true`** — allow fetching from huggingface.co. Default `true`, but explicit declaration is useful.
- **`env.allowLocalModels = false`** — don't try to read `models/` from `public/`. We have nothing in `public/`, and attempts at such fetches would litter DevTools with 404s.

In the file there's a TODO about migrating to the future `ModelRegistry`-based config API in Transformers.js (it's slated for the next major).

## HOW (progress events)

```ts
type HfProgress = {
  status?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
};

function adaptProgress(p: HfProgress, onProgress: ProgressFn) {
  let phase: ProgressPhase = 'download';
  if (p.status === 'ready' || p.status === 'done') phase = 'warmup';
  else if (p.status === 'initiate' || p.status === 'download' || p.status === 'progress')
    phase = 'download';
  else phase = 'compile';

  const pct =
    typeof p.progress === 'number'
      ? Math.max(0, Math.min(100, p.progress))
      : p.total && p.loaded
        ? (p.loaded / p.total) * 100
        : 0;

  onProgress({ phase, pct, file: p.file });
}
```

Transformers.js v4 sends progress events with different `status`'s:
- `initiate` / `download` / `progress` — network download
- `done` / `ready` — file loaded / model ready
- (others) — compilation

We adapt this into our typed `ProgressEvent { phase, pct, file? }` and send it out:

```ts
// Worker:
send({ type: 'progress', phase: p.phase, pct: p.pct, file: p.file });
```

The main thread receives `progress` messages and updates the UI:

```tsx
// pseudocode React component
{state === 'loading' && (
  <ProgressBar
    label={`Downloading model (${currentFile})...`}
    pct={pct}
  />
)}
```

The progress bar matters because 110 MB on 4G is 30+ seconds. Without visual feedback the user will decide the page hung.

## HOW (model warm-up and cold start)

When `pipeline()` finishes — the model is loaded and compiled, but **the first inference is usually 2–3× slower**, because:

1. Shader binaries are compiled lazily on the first call.
2. CPU pages of weights are lazily pulled into video memory.
3. The allocator allocates buffers for intermediate tensors.

To "warm up" — right after init you can call `ner('')` or `ner('warmup')`. In our current code this isn't there — the first "user" inference is also the warm-up. If you want honest latency metrics from the first keystroke — you should add a warm-up call inside `ensureInit`. This is a project simplification.

## Connection with the UI

In the main thread:

```ts
// pseudocode
const worker = new Worker(..., { type: 'module' });
const snapshot = new SnapshotManager({ worker, onApply: ... });

worker.addEventListener('message', (e: MessageEvent<FromWorker>) => {
  const msg = e.data;
  if (msg.type === 'progress') {
    setProgress({ phase: msg.phase, pct: msg.pct, file: msg.file });
  } else if (msg.type === 'ready') {
    setProgress(null);
    setBackend(msg.backend);
    setLoadedFrom(msg.modelLoadedFrom);
  } else if (msg.type === 'error') {
    setError(msg.message);
  }
  snapshot.handleWorkerMessage(msg); // also try as inferred event
});
```

A few UX points:

- **Show backend** ("Running on WebGPU" / "Running on WASM (slower)"). This is both debug info and a privacy aspect: the user sees that the model runs locally.
- **Show `modelLoadedFrom`** ("Loaded from cache" / "Downloaded 110 MB"). This explains why the first launch is slower.
- **Don't block the editor** — regexes work independently of ML. The user can start typing right away, will see EMAIL/PHONE/etc. highlighting, in a second ML highlighting for PERSON/LOCATION arrives.

## Failure modes

**Cache storage is full.** The browser may evict the cache (Chrome gives several GB per origin, but under overall disk pressure it clears LRU). Then on the next visit it's again 110 MB. Not critical, just slow. You can use `navigator.storage.persist()` to request persistent storage — we don't (simplification).

**HF CDN is unavailable.** Doesn't load. In our Worker the catch sends `error` out. Main thread shows a banner; regex spans keep working. You can stand up a CDN replacement if HF is down — but for a prod system that's an infrastructure question.

**WebGPU compilation fails mid-progress.** Progress showed 100% download, then silence. Try/catch catches, falls back to WASM, progress starts again for WASM-specific operations. UX: show a "WebGPU didn't work, switching to WASM..." banner.

**`progress`-events spam.** Transformers.js can send progress tens of times per second. Worker → main thread message rate. In practice not a problem (postMessage 0–1 ms), but on a slow network and many files (tokenizer.json + model.onnx + config.json + ...) you can throttle on the Worker side.

**`@huggingface/transformers` API version changes.** In v4 the runtime was rewritten, pipeline API is compatible, but `env` keys could have changed. Worth checking `node_modules/@huggingface/transformers/types/env.d.ts` after an upgrade. We have a comment in code that we write `env.useBrowserCache` — if in v5 it's renamed, a migration is needed.

**Browser fingerprinting via WebGPU.** WebGPU adapter info (vendor, device) is a potential fingerprint surface. For a privacy app like ours this is ironic: we encrypt data from the network but report a hardware profile. In practice this is already accessible via WebGL, so we don't mythologize this problem. But in a very paranoid scenario it's worth thinking about.

## What should click

1. **Backend selection — capability detect + try/catch fallback.** `'gpu' in navigator` is necessary, not sufficient.
2. **`env.useBrowserCache = true`** — one-liner that does the whole cache.
3. **Cache Storage > IndexedDB for weights** — 10× faster reads (Chrome team recommendation 2026).
4. **OPFS — for multi-GB models.** For 110 MB Cache Storage is simpler and works fine.
5. **Progress events are necessary** — without them the user sees a hung page.
6. **Cold start includes download + compile + warm-up.** All three have different nature and are worth separating for UX.

In the next chapter — the eval harness: how we even know the model works, and how we measure the WebGPU vs WASM difference.
