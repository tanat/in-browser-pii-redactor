import { useEffect, useMemo, useRef, useState } from 'react';
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

  return (
    <main className="mx-auto max-w-3xl p-8 space-y-4">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">In-Browser PII Redactor</h1>
        <a href="/eval" className="text-sm text-blue-600 hover:underline">
          /eval →
        </a>
      </header>
      <StatusBar status={status} latency={latency} />
      <div className="flex items-center gap-3">
        <label className="text-xs flex items-center gap-1">
          <input
            type="checkbox"
            checked={masked}
            onChange={(e) => setMasked(e.target.checked)}
          />
          Mask spans as [TYPE]
        </label>
      </div>
      <Editor initialText={SAMPLE} spans={merged} masked={masked} onTextChange={onTextChange} />
      <Legend />
      <details className="text-xs text-gray-600">
        <summary>Detected spans ({merged.length})</summary>
        <ul className="mt-2 space-y-1">
          {merged.map((s, i) => (
            <li key={i}>
              <code className="bg-gray-100 px-1">[{s.type}]</code> &quot;{s.text}&quot; @ {s.start}-
              {s.end} ({s.source}, conf {s.confidence.toFixed(3)})
            </li>
          ))}
        </ul>
      </details>
    </main>
  );
}

function StatusBar({ status, latency }: { status: Status; latency: number | null }) {
  const right = latency !== null ? `${latency.toFixed(0)} ms / inference` : '';
  let left: React.ReactNode;
  if (status.kind === 'init') left = 'Booting worker…';
  else if (status.kind === 'progress')
    left = `${status.phase}${status.file ? ' ' + status.file : ''}: ${status.pct.toFixed(0)}%`;
  else if (status.kind === 'ready') left = `Ready on ${status.backend}.`;
  else left = `Error: ${status.message}`;
  return (
    <div className="flex justify-between text-xs text-gray-700">
      <span>{left}</span>
      <span>{right}</span>
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
    <div className="flex flex-wrap gap-2 text-xs">
      {TYPES.map((t) => (
        <span key={t.type} className={`redaction ${t.cls}`}>
          {t.label}
        </span>
      ))}
    </div>
  );
}
