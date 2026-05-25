import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SnapshotManager, type SnapshotWorker } from '../snapshot';
import type { ToWorker, FromWorker } from '../../types/messages';

function makeWorker() {
  const sent: ToWorker[] = [];
  const worker: SnapshotWorker = {
    postMessage: (msg: ToWorker) => sent.push(msg),
  };
  return { worker, sent };
}

describe('SnapshotManager', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('case 1: rapid 5 changes → only the last reaches the worker', () => {
    const { worker, sent } = makeWorker();
    const mgr = new SnapshotManager({ worker, debounceMs: 200, onApply: vi.fn() });
    for (const t of ['a', 'ab', 'abc', 'abcd', 'abcde']) {
      mgr.onTextChange(t);
      vi.advanceTimersByTime(50); // less than 200ms
    }
    expect(sent.length).toBe(0);
    vi.advanceTimersByTime(200);
    expect(sent).toEqual([{ type: 'infer', text: 'abcde', version: 1 }]);
  });

  it('case 2: response with stale version is dropped', () => {
    const { worker } = makeWorker();
    const onApply = vi.fn();
    const mgr = new SnapshotManager({ worker, debounceMs: 200, onApply });
    mgr.onTextChange('a');
    vi.advanceTimersByTime(200); // sends v1
    mgr.onTextChange('ab');
    vi.advanceTimersByTime(200); // sends v2

    const stale: FromWorker = { type: 'inferred', spans: [], version: 1, latencyMs: 50 };
    mgr.handleWorkerMessage(stale);
    expect(onApply).not.toHaveBeenCalled();
  });

  it('case 3: response with current version is applied', () => {
    const { worker } = makeWorker();
    const onApply = vi.fn();
    const mgr = new SnapshotManager({ worker, debounceMs: 200, onApply });
    mgr.onTextChange('hello');
    vi.advanceTimersByTime(200);
    const fresh: FromWorker = { type: 'inferred', spans: [], version: 1, latencyMs: 33 };
    mgr.handleWorkerMessage(fresh);
    expect(onApply).toHaveBeenCalledWith([], 1, 33);
  });

  it('case 4: response arriving between debounce expiry and next change → applied', () => {
    const { worker } = makeWorker();
    const onApply = vi.fn();
    const mgr = new SnapshotManager({ worker, debounceMs: 200, onApply });
    mgr.onTextChange('hello');
    vi.advanceTimersByTime(200); // v1 sent
    const r1: FromWorker = { type: 'inferred', spans: [], version: 1, latencyMs: 30 };
    mgr.handleWorkerMessage(r1);
    expect(onApply).toHaveBeenCalledTimes(1);
    // Now the user types again; v1 was already applied, no double-apply.
    mgr.onTextChange('hello!');
    vi.advanceTimersByTime(200); // v2 sent
    const r2: FromWorker = { type: 'inferred', spans: [], version: 2, latencyMs: 30 };
    mgr.handleWorkerMessage(r2);
    expect(onApply).toHaveBeenCalledTimes(2);
    expect(onApply).toHaveBeenLastCalledWith([], 2, 30);
  });
});
