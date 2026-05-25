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
