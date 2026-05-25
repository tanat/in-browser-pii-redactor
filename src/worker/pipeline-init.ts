import { pipeline, type PipelineType, type PretrainedModelOptions } from '@huggingface/transformers';
import type { Backend, ProgressPhase } from '../types/messages';

export const MODEL_ID = 'Xenova/bert-base-NER';
export const TASK: PipelineType = 'token-classification';

export type ProgressEvent = { phase: ProgressPhase; pct: number; file?: string };
export type ProgressFn = (e: ProgressEvent) => void;

export type InitResult = {
  // The constructed pipeline. Typed as `unknown` because callers only invoke it.
  ner: (text: string, opts?: Record<string, unknown>) => Promise<unknown>;
  backend: Backend;
};

export function selectDevice(): Backend {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    return 'webgpu';
  }
  return 'wasm';
}

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
