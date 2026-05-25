import { useEffect, useRef, useState } from 'react';
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
    <div className="relative">
      <div
        ref={ref}
        contentEditable
        suppressContentEditableWarning
        onInput={onInput}
        onClick={onClick}
        className="min-h-32 border rounded p-3 text-sm font-mono outline-none focus:ring-2 focus:ring-blue-300 leading-relaxed whitespace-pre-wrap"
      />
      {popover && (
        <div
          className="absolute z-10 bg-white border rounded shadow text-xs p-2 max-w-sm"
          style={{ left: popover.x, top: popover.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="font-semibold">{popover.span.type}</div>
          <div className="text-gray-600">
            original: <span className="font-mono">{popover.span.text}</span>
          </div>
          <div className="text-gray-500">
            source: {popover.span.source} · confidence:{' '}
            {popover.span.confidence.toFixed(3)}
          </div>
          <button
            className="mt-1 text-blue-600 hover:underline"
            onClick={() => setPopover(null)}
          >
            close
          </button>
        </div>
      )}
    </div>
  );
}
