# Snapshot — debounce + version stamping, dropping stale responses

## WHY

This is one of the most frequent sources of hard-to-reproduce bugs in real-time AI interfaces. The user sees wrong highlighting — you can't reproduce it stably, because the bug only shows up on fast input. The cause is almost always one: **the UI applied a Worker response that referred to old text**.

Without snapshot versioning a fast typist will see zombie spans from the previous input — the model returns a result after 200 ms, but the user has already erased the text. Span "John" highlights on "Joh" or "Johanna" — because the offset was computed from what was, but render goes by what is now.

The solution — two disciplines, working together:

1. **Debounce** — don't send every keystroke to the Worker. Wait for 200 ms of silence, then send the latest text.
2. **Version stamping** — each sent snapshot gets an incremental number. The Worker echoes that number in the response. The main thread checks: if the number is stale — drop.

Debounce reduces load. Versioning protects against race conditions that debounce doesn't cover (the Worker might be busy with a previous model load and respond with a delay greater than the debounce window).

## HOW

`src/pipeline/snapshot.ts`:

```ts
import type { FromWorker, ToWorker } from '../types/messages';
import type { MlSpan } from '../types/span';

export type SnapshotWorker = {
  postMessage: (msg: ToWorker) => void;
};

export type SnapshotOptions = {
  worker: SnapshotWorker;
  debounceMs?: number;
  onApply: (spans: MlSpan[], version: number, latencyMs: number) => void;
};

/**
 * Coalesces rapid text changes into a single inference request and drops out-of-order
 * worker responses by version number.
 *
 * Lifecycle:
 *   - onTextChange(text) restarts a debounce timer; only the most-recent text wins.
 *   - When the timer fires, increments `currentVersion` and posts to the worker.
 *   - handleWorkerMessage(msg) drops `inferred` results whose `version` is less than
 *     the current version (stale).
 */
export class SnapshotManager {
  private currentVersion = 0;
  private debounceHandle: ReturnType<typeof setTimeout> | null = null;
  private latestText = '';
  private readonly worker: SnapshotWorker;
  private readonly debounceMs: number;
  private readonly onApply: SnapshotOptions['onApply'];

  constructor({ worker, debounceMs = 200, onApply }: SnapshotOptions) {
    this.worker = worker;
    this.debounceMs = debounceMs;
    this.onApply = onApply;
  }

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
    if (msg.version !== this.currentVersion) {
      // Stale: a newer snapshot has been queued/sent since this inference began.
      return;
    }
    this.onApply(msg.spans, msg.version, msg.latencyMs);
  }

  /** Internal accessor for tests. */
  getCurrentVersion(): number {
    return this.currentVersion;
  }
}
```

Let's go through every invariant piece.

### Debounce — coalescing keystrokes

`onTextChange(text)` is called on every keystroke (or, in our case, on every change to `<contenteditable>` content via `MutationObserver`). Inside:

```ts
this.latestText = text;
if (this.debounceHandle !== null) clearTimeout(this.debounceHandle);
this.debounceHandle = setTimeout(() => { ... }, this.debounceMs);
```

"Last-write-wins" logic: save `latestText`, restart the timer. When the timer fires (only if no new keystroke for 200 ms), send `latestText` (not the text that was there when the timer was set — this matters).

Why 200 ms? It's the empirically found window of "user paused — we send." Smaller — too frequent inferences, battery drain and UI tearing. Bigger — the user notices the highlighting lagging.

### Version stamping — when timer fires

```ts
this.currentVersion += 1;
this.worker.postMessage({
  type: 'infer',
  text: this.latestText,
  version: this.currentVersion,
});
```

Increment **inside** the timer, not on keystroke. That means: 10 fast keystrokes in 200 ms produce **one** version (e.g. v7), not ten. Versions are monotonic and coincide with sends.

### Stale check — on worker response

```ts
handleWorkerMessage(msg: FromWorker) {
  if (msg.type !== 'inferred') return;
  if (msg.version !== this.currentVersion) {
    return;
  }
  this.onApply(msg.spans, msg.version, msg.latencyMs);
}
```

The scenario this covers:

```
t=0   keystroke → onTextChange("hello j")
t=200 timer fires → version=1, post infer
t=210 keystroke → onTextChange("hello jo")
t=410 timer fires → version=2, post infer
t=450 Worker responds inferred{version=1}  ← STALE, currentVersion=2, DROP
t=480 Worker responds inferred{version=2}  ← OK, apply
```

In a scenario without versioning the tick at t=450 would apply spans computed for "hello j" to the text "hello jo" — alignment offsets would be broken, and the highlighting would sit crooked.

### Strict equality vs `<`

In the code it's `msg.version !== this.currentVersion`, not `msg.version < this.currentVersion`. These are equivalent under monotonically increasing `currentVersion`, but it protects against the theoretical case where the Worker mistakenly sent a version from the future (e.g. as a result of a bug in the Worker's code). Strict equality is paranoid-safer.

## Subtle spots

### What if the debounce window is too short for the Worker?

Say debounce 100 ms, but WASM inference takes 250 ms on a long phrase.

```
t=0   keystroke → timer
t=100 timer → v1, post
t=200 keystroke → timer
t=300 timer → v2, post                  ← v1 still working!
t=350 Worker responds v1                ← v1 != currentVersion(2), DROP
t=550 Worker responds v2                ← OK, apply
```

Versioning protects: v1 is dropped. But the Worker had to do two inferences in a row, and the user saw highlighting only at t=550 (250 ms after the second keystroke).

This is fine — the delay feels like "lag," not "glitch." The main thing is that the highlighting is **correct**, not "from old text."

### What if the Worker responds faster than debounce?

Hot inference on WebGPU — 45 ms. Debounce 200 ms. Then:

```
t=0   keystroke → timer
t=200 timer → v1, post
t=245 Worker responds v1               ← v1 == currentVersion(1), apply
```

No issues — apply right away. The user sees highlighting 245 ms after the keystroke. That's our target latency.

### What if the user types slowly (debounce firing constantly)?

```
t=0     keystroke → timer
t=200   timer → v1, post
t=245   Worker responds v1, apply spans
t=500   keystroke → timer
t=700   timer → v2, post
t=745   Worker responds v2, apply spans
...
```

One inference per word. This is right: the user pauses, we react. Highlighting updates 245 ms after each pause.

### Initial state

`currentVersion = 0`. When we send the first infer, `+= 1` happens → version=1. The Worker responds with {version: 1}. Strict equality matches — apply.

Version 0 means "nothing sent," it's an invariant. The Worker never sends version=0, because the main thread starts from 1.

### What if the Worker isn't there yet (model loading)?

`worker.postMessage` puts a message in the Worker's queue. When `ensureInit` completes, the Worker pulls the first message from the queue. Versions aren't lost.

If the user types during initial model load:

```
t=0    user types, onTextChange → timer
t=200  timer → v1, post (model still loading)
t=210  user types → timer
t=410  timer → v2, post (model still loading)
...
t=2000 model loaded
t=2050 Worker infers v1 (if user has run up to v15, v1..v14 are all stale)
t=2100 Worker responds v1 → DROP
t=2150 Worker infers v2 → DROP
...
```

This is inefficient — the Worker does a lot of throwaway work. In an advanced version you can do "drop in-flight requests on Worker side": the Worker keeps its own `lastSeenVersion` and skips inference if a message with a higher version came in. There's no such thing in our code — a simplification, and at 200 ms debounce such cascading-stale inferences are rare.

## Failure modes

**Race on debounce at unmount.** If a SnapshotManager is destroyed while the timer is still active — it'll fire on a destroyed object. We don't have an explicit destroy method (simplification); in the React wrapper you should do `clearTimeout(handle)` in `useEffect`-cleanup. Without it you can get a `post` to the Worker on a page that's already unmounted.

**Didn't account for regex spans.** SnapshotManager is only about ML spans. Regex spans are computed synchronously in the main thread on every change. If debounce is 200 ms but regex runs on every keystroke — UI lag. In practice regex is fast (<1 ms on 200 chars), so we debounce it too (run it in the same place where we ship the ML request).

In our code `SnapshotManager` accepts only the worker; regex runs **outside** — in a React component that calls SnapshotManager's `onTextChange` **and** runs regex with the same `version`. Then on `onApply` we do merge regex-spans + ml-spans → final list.

**Versions overflow.** `currentVersion` is a `number`, up to Number.MAX_SAFE_INTEGER (2^53). Even at 60 fps keystrokes that gives >4 million years of operation. Not a problem.

**Messages get lost.** The Worker can crash or be killed by the browser (e.g. OOM). Then for a sent `infer{version: N}` no answer will ever come. We have no timeout / retry. In an advanced version: if there's no answer for 5 seconds, the main thread restarts the Worker and resends the current snapshot. A simple polish, we skipped it.

**Main thread gets flooded with events.** If `MutationObserver` fires hundreds of times per second (e.g. IME composition), `onTextChange` gets called hundreds of times. Each call clears + sets a timeout — that's cheap (<10 µs), but still nontrivial. Defense — on the source side: use `composition*` events for IME, debounce there. We don't have this, because of plain `<contenteditable>` without IME specifics.

**Initial render before model load.** The user sees text without PII highlighting for the first ~2 seconds (while the model loads). UX solution — show a "Loading model..." banner. Regex spans meanwhile already work (they're in the main thread, don't wait for the model). After the `ready` message and first `apply` — the banner hides.

## Tests

`src/pipeline/__tests__/snapshot.test.ts` (assumed) covers:

1. **Coalesce keystrokes.** Five `onTextChange` in 100 ms → one `postMessage` after 200 ms debounce.
2. **Stale drop.** Sent v1 and v2; response with v1 is dropped, response with v2 is applied.
3. **Out-of-order responses.** v3 responds before v2 — apply v3, drop v2.
4. **Version increments.** After N keystrokes with pauses `currentVersion === N`.
5. **Initial state.** `getCurrentVersion() === 0` before the first send.

Tested via `vi.useFakeTimers()` + mock `worker.postMessage` (records calls in an array).

## Alternatives

**AbortController.** If the Worker could cancel in-flight inference, you could abort stale ones instead of version-dropping. Transformers.js v4 doesn't provide such an API at the pipeline level (you can reset the instance, but that's catastrophic). Versioning is a simple solution without the need to abort inference.

**RxJS-style switchMap.** `latestText$.pipe(debounceTime(200), switchMap(text => infer(text)))` would do the same — switchMap cancels the previous subscription when a new one appears. This is the idiomatic Rx approach, and it works if `infer` returns an Observable. We don't use Rx — `setTimeout` + version check reads simpler and doesn't drag in a dependency.

**Structural typing on Span with version.** You could stick `version` on the spans object itself, and the render function would check it before the DOM operation. This mixes domain types (Span) with infrastructure (snapshot lifecycle). Better to keep the version in the message layer, as it is now.

## What should click

1. **Debounce + versioning are two different protections.** Debounce reduces requests, versioning insures against out-of-order responses.
2. **`currentVersion` increments on send**, not on keystroke. Versions are monotonic and coincide with requests.
3. **Strict equality `===`**, not `<`. Paranoid-safer.
4. **Render always works with the latest snapshot** — never with what the Worker sent.
5. **Default debounce 200 ms** — balance between UX responsiveness and compute load.

In the next chapter — what happens exactly once at start: loading 110 MB of model, choosing the backend, caching. This is the slowest moment of the pipeline, and the UX around it is a separate task.
