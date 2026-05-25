import { useState } from 'react';
import { runBenchmark, type BenchmarkReport } from '../eval/benchmark';
import type { Fixture, RunReport } from '../eval/harness';
import fixtures from '../fixtures/synthetic.json';
import { readAll, clearLog, downloadNdjson } from '../storage/ndjson-log';

const FIXTURES = fixtures as unknown as Fixture[];

export function EvalDashboard() {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number; label: string } | null>(
    null,
  );

  const onRun = async () => {
    setRunning(true);
    setReport(null);
    try {
      const r = await runBenchmark(FIXTURES, (done, total, label) =>
        setProgress({ done, total, label }),
      );
      setReport(r);
    } catch (e) {
      console.error(e);
    } finally {
      setRunning(false);
      setProgress(null);
    }
  };

  const onDownload = () => {
    if (!report) return;
    const json = JSON.stringify([report.webgpu, report.wasm].filter(Boolean), null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'eval-results.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const onExportLog = async () => {
    const recs = await readAll();
    downloadNdjson(recs);
  };

  return (
    <main className="mx-auto max-w-4xl p-8 space-y-6">
      <h1 className="text-2xl font-bold">Evaluation</h1>
      <p className="text-sm text-gray-600">
        Runs all {FIXTURES.length} fixtures through the full pipeline (regex + ML + merge),
        scored as span-level F1 with IoU ≥ 0.5. Each fixture is run on both WebGPU and WASM.
      </p>
      <div className="flex gap-3">
        <button
          onClick={onRun}
          disabled={running}
          className="px-3 py-1 rounded bg-black text-white disabled:opacity-50"
        >
          {running ? 'Running…' : 'Run evaluation'}
        </button>
        <button
          onClick={onDownload}
          disabled={!report}
          className="px-3 py-1 rounded border disabled:opacity-50"
        >
          Download eval-results.json
        </button>
        <button onClick={onExportLog} className="px-3 py-1 rounded border">
          Export inference log (NDJSON)
        </button>
        <button
          onClick={() => clearLog()}
          className="px-3 py-1 rounded border text-red-600"
        >
          Clear log
        </button>
      </div>
      {progress && (
        <p className="text-sm text-gray-600">
          {progress.done}/{progress.total} — {progress.label}
        </p>
      )}
      {report && <ReportTable report={report} />}
    </main>
  );
}

function ReportTable({ report }: { report: BenchmarkReport }) {
  const types = new Set<string>();
  if (report.webgpu) Object.keys(report.webgpu.aggregate.perType).forEach((t) => types.add(t));
  Object.keys(report.wasm.aggregate.perType).forEach((t) => types.add(t));
  return (
    <div className="space-y-6">
      <h2 className="font-semibold">Aggregate</h2>
      <table className="text-sm border-collapse w-full">
        <thead>
          <tr className="bg-gray-100 text-left">
            <th className="border p-1">Backend</th>
            <th className="border p-1">P</th>
            <th className="border p-1">R</th>
            <th className="border p-1">F1</th>
            <th className="border p-1">p50 ms</th>
            <th className="border p-1">p95 ms</th>
          </tr>
        </thead>
        <tbody>
          {[report.webgpu, report.wasm]
            .filter((x): x is RunReport => Boolean(x))
            .map((r) => (
              <tr key={r.backend}>
                <td className="border p-1">{r.backend}</td>
                <td className="border p-1">{r.aggregate.precision.toFixed(3)}</td>
                <td className="border p-1">{r.aggregate.recall.toFixed(3)}</td>
                <td className="border p-1">{r.aggregate.f1.toFixed(3)}</td>
                <td className="border p-1">{r.aggregate.p50LatencyMs.toFixed(0)}</td>
                <td className="border p-1">{r.aggregate.p95LatencyMs.toFixed(0)}</td>
              </tr>
            ))}
        </tbody>
      </table>

      <h2 className="font-semibold">Per type (WASM)</h2>
      <table className="text-sm border-collapse w-full">
        <thead>
          <tr className="bg-gray-100 text-left">
            <th className="border p-1">Type</th>
            <th className="border p-1">P</th>
            <th className="border p-1">R</th>
            <th className="border p-1">F1</th>
            <th className="border p-1">TP/FP/FN</th>
          </tr>
        </thead>
        <tbody>
          {Array.from(types)
            .sort()
            .map((t) => {
              const m = report.wasm.aggregate.perType[t] ?? {
                precision: 0,
                recall: 0,
                f1: 0,
                tp: 0,
                fp: 0,
                fn: 0,
              };
              return (
                <tr key={t}>
                  <td className="border p-1">{t}</td>
                  <td className="border p-1">{m.precision.toFixed(3)}</td>
                  <td className="border p-1">{m.recall.toFixed(3)}</td>
                  <td className="border p-1">{m.f1.toFixed(3)}</td>
                  <td className="border p-1">
                    {m.tp}/{m.fp}/{m.fn}
                  </td>
                </tr>
              );
            })}
        </tbody>
      </table>

      <h2 className="font-semibold">Per fixture (WASM)</h2>
      <table className="text-sm border-collapse w-full">
        <thead>
          <tr className="bg-gray-100 text-left">
            <th className="border p-1">Fixture</th>
            <th className="border p-1">F1</th>
            <th className="border p-1">P / R</th>
            <th className="border p-1">Latency</th>
          </tr>
        </thead>
        <tbody>
          {report.wasm.perFixture.map((f) => (
            <tr key={f.id}>
              <td className="border p-1 font-mono">{f.id}</td>
              <td className="border p-1">{f.metrics.f1.toFixed(3)}</td>
              <td className="border p-1">
                {f.metrics.precision.toFixed(2)} / {f.metrics.recall.toFixed(2)}
              </td>
              <td className="border p-1">{f.latencyMs.toFixed(0)} ms</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
