import { runHarness, type Fixture, type RunReport, type HarnessProgress } from './harness';

export type BenchmarkReport = {
  webgpu: RunReport | null;
  wasm: RunReport;
};

export async function runBenchmark(
  fixtures: Fixture[],
  onProgress?: HarnessProgress,
): Promise<BenchmarkReport> {
  let webgpu: RunReport | null = null;
  // Try WebGPU first; gracefully fall through if unsupported.
  try {
    webgpu = await runHarness(fixtures, 'webgpu', (d, t, l) =>
      onProgress?.(d, t, `webgpu/${l}`),
    );
  } catch (err) {
    console.warn('[benchmark] WebGPU run failed, continuing with WASM only:', err);
  }
  const wasm = await runHarness(fixtures, 'wasm', (d, t, l) =>
    onProgress?.(d, t, `wasm/${l}`),
  );
  return { webgpu, wasm };
}
