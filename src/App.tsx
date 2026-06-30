import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ShieldCheck,
  Lock,
  Cpu,
  WifiOff,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Gauge,
  EyeOff,
  Eye,
  ChevronRight,
} from 'lucide-react';
import type { FromWorker } from './types/messages';
import type { Span } from './types/span';
import { Editor } from './components/Editor';
import { EvalDashboard } from './components/EvalDashboard';
import { extractAllRegex } from './pipeline/regex';
import { mergeSpans } from './pipeline/merge';
import { SnapshotManager } from './pipeline/snapshot';
import { logInference } from './storage/ndjson-log';

type Status =
  | { kind: 'init' }
  | { kind: 'progress'; phase: string; pct: number; file?: string }
  | { kind: 'ready'; backend: string }
  | { kind: 'error'; message: string };

const SAMPLE = `Schedule follow-up with John Smith (DOB 1962-03-14, MRN 4471832).
Reach him at john.smith@example.com or 415-555-0143.
Patient lives at 142 Maple St, Burlington VT.`;

export default function App() {
  if (typeof window !== 'undefined' && window.location.pathname === '/eval') {
    return <EvalDashboard />;
  }
  return <Main />;
}

function Main() {
  const workerRef = useRef<Worker | null>(null);
  const snapshotRef = useRef<SnapshotManager | null>(null);
  const [status, setStatus] = useState<Status>({ kind: 'init' });
  const [text, setText] = useState(SAMPLE);
  const [mlSpans, setMlSpans] = useState<Span[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [masked, setMasked] = useState(false);
  const backendRef = useRef<string>('unknown');
  const textRef = useRef(SAMPLE);

  const regexSpans = useMemo(() => extractAllRegex(text), [text]);
  const merged = useMemo(() => mergeSpans(regexSpans, mlSpans), [regexSpans, mlSpans]);

  useEffect(() => {
    const worker = new Worker(new URL('./worker/inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;
    const mgr = new SnapshotManager({
      worker,
      debounceMs: 200,
      onApply: (spans, version, latencyMs) => {
        setMlSpans(spans);
        setLatency(latencyMs);
        logInference({
          ts: Date.now(),
          textLength: textRef.current.length,
          spansCount: spans.length,
          latencyMs,
          backend: backendRef.current,
          version,
        }).catch((err) => console.warn('[log] failed', err));
      },
    });
    snapshotRef.current = mgr;
    worker.addEventListener('message', (e: MessageEvent<FromWorker>) => {
      const msg = e.data;
      if (msg.type === 'progress') {
        setStatus({ kind: 'progress', phase: msg.phase, pct: msg.pct, file: msg.file });
      } else if (msg.type === 'ready') {
        backendRef.current = msg.backend;
        setStatus({ kind: 'ready', backend: msg.backend });
        mgr.onTextChange(SAMPLE); // kick off first inference on the seeded text
      } else if (msg.type === 'inferred') {
        mgr.handleWorkerMessage(msg);
      } else if (msg.type === 'error') {
        setStatus({ kind: 'error', message: msg.message });
      }
    });
    worker.postMessage({ type: 'init' });
    return () => worker.terminate();
  }, []);

  const onTextChange = (next: string) => {
    setText(next);
    textRef.current = next;
    snapshotRef.current?.onTextChange(next);
  };

  const counts = useMemo(() => {
    const c = new Map<string, number>();
    for (const s of merged) c.set(s.type, (c.get(s.type) ?? 0) + 1);
    return c;
  }, [merged]);

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6 sm:py-12">
      <SiteHeader />

      <section className="mt-6 overflow-hidden rounded-2xl border border-slate-200 bg-white/70 shadow-sm ring-1 ring-black/[0.02] backdrop-blur-sm">
        <div className="flex flex-col gap-3 border-b border-slate-100 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-5">
          <StatusBar status={status} latency={latency} />
          <MaskToggle masked={masked} onChange={setMasked} disabled={status.kind !== 'ready'} />
        </div>

        <div className="px-4 py-4 sm:px-5 sm:py-5">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
              Your text
            </span>
            <span className="text-[0.7rem] text-slate-400">
              Click any highlight for details
            </span>
          </div>
          <Editor
            initialText={SAMPLE}
            spans={merged}
            masked={masked}
            onTextChange={onTextChange}
          />

          <EntitySummary total={merged.length} counts={counts} ready={status.kind === 'ready'} />
        </div>

        <div className="border-t border-slate-100 bg-slate-50/60 px-4 py-3 sm:px-5">
          <Legend />
        </div>
      </section>

      <details className="group mt-4 rounded-xl border border-slate-200 bg-white/60 px-4 py-3 text-xs text-slate-600 backdrop-blur-sm">
        <summary className="flex cursor-pointer list-none items-center gap-2 font-medium text-slate-700 select-none">
          <ChevronRight className="size-3.5 text-slate-400 transition-transform group-open:rotate-90" />
          Inspect detected spans
          <span className="ml-1 rounded-full bg-slate-100 px-2 py-0.5 text-[0.65rem] font-semibold text-slate-500">
            {merged.length}
          </span>
        </summary>
        {merged.length === 0 ? (
          <p className="mt-3 text-slate-400">No spans detected yet.</p>
        ) : (
          <ul className="mt-3 space-y-1.5">
            {merged.map((s, i) => (
              <li key={i} className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                <code
                  className={`redaction redaction-${s.type.toLowerCase()} !cursor-default text-[0.65rem]`}
                >
                  {s.type}
                </code>
                <span className="font-mono text-slate-700">&quot;{s.text}&quot;</span>
                <span className="text-slate-400">
                  @ {s.start}-{s.end} · {s.source} · conf {s.confidence.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </details>

      <footer className="mt-8 flex items-center justify-center gap-1.5 text-center text-[0.7rem] text-slate-400">
        <Lock className="size-3" />
        Text never leaves this tab. No network, no servers, no storage.
      </footer>
    </div>
  );
}

function SiteHeader() {
  return (
    <header>
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex size-11 items-center justify-center rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 text-white shadow-lg shadow-sky-500/20">
            <ShieldCheck className="size-6" strokeWidth={2.25} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
              In-Browser PII Redactor
            </h1>
            <p className="text-sm text-slate-500">
              Detect &amp; mask personal data — fully on-device
            </p>
          </div>
        </div>
        <a
          href="/eval"
          className="hidden shrink-0 items-center gap-1 rounded-lg border border-slate-200 bg-white/70 px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:text-slate-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 sm:inline-flex"
        >
          <Gauge className="size-3.5" />
          Eval
        </a>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        <TrustPill icon={<Cpu className="size-3.5" />} text="Runs 100% in your browser" />
        <TrustPill icon={<WifiOff className="size-3.5" />} text="Nothing leaves your device" />
        <TrustPill icon={<Lock className="size-3.5" />} text="No data is ever uploaded" />
      </div>
    </header>
  );
}

function TrustPill({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200/70 bg-emerald-50 px-2.5 py-1 text-[0.72rem] font-medium text-emerald-700">
      <span className="text-emerald-500">{icon}</span>
      {text}
    </span>
  );
}

function MaskToggle({
  masked,
  onChange,
  disabled,
}: {
  masked: boolean;
  onChange: (v: boolean) => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!masked)}
      disabled={disabled}
      aria-pressed={masked}
      className={`inline-flex shrink-0 items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:cursor-not-allowed disabled:opacity-50 ${
        masked
          ? 'border-slate-900 bg-slate-900 text-white hover:bg-slate-800'
          : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300'
      }`}
    >
      {masked ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
      {masked ? 'Masked' : 'Mask PII'}
    </button>
  );
}

function EntitySummary({
  total,
  counts,
  ready,
}: {
  total: number;
  counts: Map<string, number>;
  ready: boolean;
}) {
  if (!ready) return null;
  if (total === 0) {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-lg border border-emerald-100 bg-emerald-50/70 px-3 py-2 text-xs text-emerald-700">
        <CheckCircle2 className="size-4 shrink-0" />
        No PII detected in this text.
      </div>
    );
  }
  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-xs">
      <span className="inline-flex items-center gap-1.5 font-semibold text-slate-700">
        <AlertTriangle className="size-3.5 text-amber-500" />
        {total} {total === 1 ? 'entity' : 'entities'} found
      </span>
      <span className="text-slate-300">·</span>
      {TYPES.filter((t) => counts.has(t.type)).map((t) => (
        <span key={t.type} className={`legend-chip redaction-${t.type.toLowerCase()}`}>
          <span className="legend-dot" />
          {t.label}
          <span className="font-bold opacity-70">{counts.get(t.type)}</span>
        </span>
      ))}
    </div>
  );
}

function StatusBar({ status, latency }: { status: Status; latency: number | null }) {
  if (status.kind === 'progress') {
    const pct = Math.max(0, Math.min(100, status.pct));
    return (
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
          <Loader2 className="size-3.5 shrink-0 animate-spin text-sky-500" />
          <span className="truncate">
            Loading on-device model
            <span className="text-slate-400"> · one-time ~110 MB, cached after</span>
          </span>
          <span className="ml-auto shrink-0 tabular-nums text-slate-400">
            {pct.toFixed(0)}%
          </span>
        </div>
        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-gradient-to-r from-sky-500 to-indigo-500 transition-[width] duration-200"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  }

  let dot: React.ReactNode;
  let label: React.ReactNode;
  if (status.kind === 'init') {
    dot = <Loader2 className="size-3.5 animate-spin text-slate-400" />;
    label = <span className="text-slate-500">Booting on-device engine…</span>;
  } else if (status.kind === 'ready') {
    dot = <CheckCircle2 className="size-3.5 text-emerald-500" />;
    label = (
      <span className="text-slate-600">
        Model ready
        <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-slate-500">
          {status.backend}
        </span>
      </span>
    );
  } else {
    dot = <AlertTriangle className="size-3.5 text-red-500" />;
    label = <span className="text-red-600">Error: {status.message}</span>;
  }

  return (
    <div className="flex min-w-0 flex-1 items-center gap-2 text-xs font-medium">
      {dot}
      <span className="truncate">{label}</span>
      {status.kind === 'ready' && latency !== null && (
        <span className="ml-auto inline-flex shrink-0 items-center gap-1 tabular-nums text-slate-400">
          <Gauge className="size-3" />
          {latency.toFixed(0)} ms
        </span>
      )}
    </div>
  );
}

const TYPES: { type: string; label: string; cls: string }[] = [
  { type: 'EMAIL', label: 'Email', cls: 'redaction-email' },
  { type: 'PHONE', label: 'Phone', cls: 'redaction-phone' },
  { type: 'SSN', label: 'SSN', cls: 'redaction-ssn' },
  { type: 'DATE', label: 'Date', cls: 'redaction-date' },
  { type: 'ADDRESS', label: 'Address', cls: 'redaction-address' },
  { type: 'PERSON', label: 'Person', cls: 'redaction-person' },
  { type: 'LOCATION', label: 'Location', cls: 'redaction-location' },
];

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-2">
      <span className="text-[0.7rem] font-medium uppercase tracking-wide text-slate-400">
        Legend
      </span>
      {TYPES.map((t) => (
        <span key={t.type} className={`legend-chip ${t.cls}`}>
          <span className="legend-dot" />
          {t.label}
        </span>
      ))}
    </div>
  );
}
