import { env } from '@huggingface/transformers';

/**
 * Configure Transformers.js to cache models in browser storage.
 *
 * Transformers.js v4 caches models via the Cache Storage API by default in browsers.
 * It also exposes `env.useFSCache` / `env.cacheDir` as knobs. OPFS isn't a separately
 * selectable backend in v4 — the library prefers `caches` (Cache Storage), which is
 * file-system-backed in Chromium and behaves similarly to OPFS for read latency.
 *
 * This function ensures caching is enabled and is a single place to swap in a more
 * specific OPFS implementation later if needed.
 */
export function setupOpfsCache(): void {
  // Enable browser cache (Cache Storage API) for model files.
  // TODO: migrate to ModelRegistry-based config in a future Transformers.js refactor.
  env.useBrowserCache = true;
  // Allow remote model fetching on first run; subsequent runs hit the cache.
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
}
