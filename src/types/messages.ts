import type { MlSpan } from './span';

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
