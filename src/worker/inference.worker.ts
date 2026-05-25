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

export {};
