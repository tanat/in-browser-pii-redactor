import { describe, it, expect, vi, afterEach } from 'vitest';
import { selectDevice } from '../pipeline-init';

describe('selectDevice', () => {
  const original = (globalThis as { navigator?: Navigator }).navigator;

  afterEach(() => {
    if (original === undefined) delete (globalThis as { navigator?: Navigator }).navigator;
    else (globalThis as { navigator?: Navigator }).navigator = original;
    vi.restoreAllMocks();
  });

  it("returns 'webgpu' when 'gpu' is on navigator", () => {
    (globalThis as unknown as { navigator: { gpu: unknown } }).navigator = { gpu: {} };
    expect(selectDevice()).toBe('webgpu');
  });

  it("returns 'wasm' when 'gpu' is not on navigator", () => {
    (globalThis as unknown as { navigator: object }).navigator = {};
    expect(selectDevice()).toBe('wasm');
  });

  it("returns 'wasm' when navigator is undefined", () => {
    // jsdom always defines navigator; emulate by deleting then restoring after.
    delete (globalThis as { navigator?: Navigator }).navigator;
    expect(selectDevice()).toBe('wasm');
  });
});
