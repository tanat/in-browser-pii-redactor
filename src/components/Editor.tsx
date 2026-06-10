import { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { charRangeToDomRange } from '../alignment/char-to-dom';
import type { Span } from '../types/span';

type EditorProps = {
  initialText: string;
  spans: Span[];
  masked: boolean;
  onTextChange: (text: string) => void;
};

type Popover = {
  span: Span;
  x: number;
  y: number;
};

export function Editor({ initialText, spans, masked, onTextChange }: EditorProps) {
  const ref = useRef<HTMLDivElement>(null);
  const lastAppliedRef = useRef<string>('');
  const initializedRef = useRef(false);
  const seenRef = useRef<Set<string>>(new Set());
  const [popover, setPopover] = useState<Popover | null>(null);

  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;
    if (ref.current) {
      ref.current.textContent = initialText;
      onTextChange(initialText);
    }
  }, [initialText, onTextChange]);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const text = root.textContent ?? '';
    const key = JSON.stringify({ masked, list: spans.map((s) => [s.type, s.start, s.end, s.text]) });
    if (key === lastAppliedRef.current) return;
    lastAppliedRef.current = key;

    root.textContent = text;
    setPopover(null);

    const sorted = [...spans].sort((a, b) => a.start - b.start);
    for (let i = sorted.length - 1; i >= 0; i--) {
      const s = sorted[i];
      try {
        const range = charRangeToDomRange(root, s.start, s.end);
        const wrap = document.createElement('span');
        wrap.className = `redaction redaction-${s.type.toLowerCase()}`;
        wrap.dataset.type = s.type;
        wrap.dataset.source = s.source;
        wrap.dataset.start = String(s.start);
        wrap.dataset.end = String(s.end);
        wrap.dataset.confidence = s.confidence.toFixed(3);
        wrap.dataset.original = s.text;
        const fingerprint = `${s.type}:${s.start}:${s.end}:${s.text}`;
        const isNew = !seenRef.current.has(fingerprint);
        if (isNew) {
          wrap.classList.add('redaction-pulse');
          seenRef.current.add(fingerprint);
        }
        if (masked) {
          wrap.textContent = `[${s.type}]`;
        } else {
          try {
            range.surroundContents(wrap);
          } catch {
            const frag = range.extractContents();
            wrap.appendChild(frag);
            range.insertNode(wrap);
          }
          continue;
        }
        // masked branch: replace original content with the wrap node holding [TYPE].
        try {
          range.deleteContents();
          range.insertNode(wrap);
        } catch (err) {
          console.warn('[Editor] mask insert failed', s, err);
        }
      } catch (err) {
        console.warn('[Editor] failed to highlight span', s, err);
      }
    }
  }, [spans, masked]);

  const onInput = () => {
    if (!ref.current) return;
    const text = ref.current.textContent ?? '';
    lastAppliedRef.current = '';
    setPopover(null);
    onTextChange(text);
  };

  const onClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    const wrap = target.closest('.redaction') as HTMLElement | null;
    if (!wrap) {
      setPopover(null);
      return;
    }
    const rect = wrap.getBoundingClientRect();
    const containerRect = ref.current?.getBoundingClientRect();
    setPopover({
      span: {
        type: wrap.dataset.type as Span['type'],
        start: Number(wrap.dataset.start),
        end: Number(wrap.dataset.end),
        text: wrap.dataset.original ?? '',
        source: (wrap.dataset.source as Span['source']) ?? 'ml',
        confidence: Number(wrap.dataset.confidence ?? 0),
      },
      x: rect.left - (containerRect?.left ?? 0),
      y: rect.bottom - (containerRect?.top ?? 0) + 4,
    });
  };

  return (
    <div className="group relative">
      <div
        ref={ref}
        contentEditable
        suppressContentEditableWarning
        spellCheck={false}
        onInput={onInput}
        onClick={onClick}
        role="textbox"
        aria-multiline="true"
        aria-label="Text to redact"
        className="min-h-48 w-full rounded-xl border border-slate-200 bg-white/90 p-4 text-[0.9rem] font-mono leading-7 text-slate-800 shadow-sm outline-none transition focus:border-sky-400 focus:shadow-[0_0_0_4px_rgba(56,189,248,0.16)] sm:p-5 sm:text-sm"
      />
      {popover && (
        <div
          className="absolute z-20 max-w-xs rounded-xl border border-slate-200 bg-white/95 p-3 text-xs shadow-xl ring-1 ring-black/5 backdrop-blur"
          style={{ left: popover.x, top: popover.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between gap-3">
            <span
              className={`redaction redaction-${popover.span.type.toLowerCase()} text-[0.7rem]`}
            >
              {popover.span.type}
            </span>
            <button
              className="rounded-md p-0.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              onClick={() => setPopover(null)}
              aria-label="Close details"
            >
              <X className="size-3.5" />
            </button>
          </div>
          <div className="mt-2 break-all font-mono text-[0.78rem] font-medium text-slate-800">
            {popover.span.text}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[0.68rem] text-slate-500">
            <span className="inline-flex items-center gap-1">
              <span className="font-medium uppercase tracking-wide text-slate-400">
                source
              </span>
              {popover.span.source}
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="font-medium uppercase tracking-wide text-slate-400">
                conf
              </span>
              {popover.span.confidence.toFixed(3)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
