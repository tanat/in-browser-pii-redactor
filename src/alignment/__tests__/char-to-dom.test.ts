import { describe, it, expect, beforeEach } from 'vitest';
import { charRangeToDomRange } from '../char-to-dom';

function makeRoot(html: string): HTMLDivElement {
  const root = document.createElement('div');
  root.innerHTML = html;
  document.body.appendChild(root);
  return root;
}

describe('charRangeToDomRange', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('case 1: single text node, range in the middle', () => {
    const root = makeRoot('Hello world');
    const r = charRangeToDomRange(root, 6, 11);
    expect(r.toString()).toBe('world');
  });

  it('case 2: range spans 2 text nodes', () => {
    const root = makeRoot('');
    root.appendChild(document.createTextNode('Hello '));
    root.appendChild(document.createTextNode('world'));
    const r = charRangeToDomRange(root, 0, 11);
    expect(r.toString()).toBe('Hello world');
    // Start should be in first text node, end in second.
    expect((r.startContainer as Text).data).toBe('Hello ');
    expect((r.endContainer as Text).data).toBe('world');
  });

  it('case 3: text + <br> interleaving', () => {
    const root = makeRoot('John<br>Smith');
    // Plain-text content (text-node concatenation) is "JohnSmith" — 9 chars.
    const r = charRangeToDomRange(root, 0, 9);
    expect(r.toString()).toBe('JohnSmith');
  });

  it('case 4: text with nested <span>', () => {
    const root = makeRoot('Hello <span>brave</span> world');
    // Concatenation: "Hello brave world"
    const r = charRangeToDomRange(root, 6, 11);
    expect(r.toString()).toBe('brave');
    expect((r.startContainer as Text).data).toBe('brave');
  });

  it('case 5: empty text node interleaved', () => {
    const root = makeRoot('');
    root.appendChild(document.createTextNode('abc'));
    root.appendChild(document.createTextNode(''));
    root.appendChild(document.createTextNode('def'));
    const r = charRangeToDomRange(root, 2, 5);
    expect(r.toString()).toBe('cde');
  });

  it('case 6: end offset past total length is clamped', () => {
    const root = makeRoot('short');
    const r = charRangeToDomRange(root, 0, 999);
    expect(r.toString()).toBe('short');
    // End must land at the end of the last text node.
    expect(r.endOffset).toBe(5);
  });
});
