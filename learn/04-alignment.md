# Alignment — BPE → char → DOM Range (the main exercise)

## WHY

All previous chapters were about infrastructure: privacy, Worker, hybrid pipeline. This one is about the central algorithm of the project. Without it nothing highlights in the editor, and the whole pipeline turns into "we have spans somewhere, but we don't know where."

The problem is stated thus. You have:

1. **The source text** in a `<contenteditable>`. From JS's standpoint it's an HTML tree: text nodes, interleaved with `<span>`, `<br>`, etc.
2. **Raw BERT NER output**: a list of objects `{ entity_group: 'PER', score: 0.998, word: 'John Smith' }`. **Without offsets** — Transformers.js v4 doesn't provide them.
3. **You need**: for each span, find the exact DOM Range so you can wrap it in `<mark>` or replace with a badge.

Between (2) and (3) there are two independent problems:

**BPE → char problem.** BERT works with BPE tokens (`John`, `##son`, `Smith`). After inference Transformers.js aggregates them into word-level entities via `aggregation_strategy: 'simple'` — but in the result the character-level alignment with the source text is lost. The decoded `word` may differ from the original substring: lowercased for uncased models, extra/removed spaces, especially around punctuation.

**char → DOM Range problem.** Say we know the span sits on `text.slice(27, 38)`. But in the DOM that text may be split across several text nodes due to nested tags. Creating a Range requires **specifying a concrete Text node + offset within it**, not "27 in the plain-text projection."

Both problems are solved by simple but subtle algorithms. And they're the best thing in this project to show in an interview. "I work with models" everybody says, "I aligned BPE tokens with the DOM" — nobody, because it's rarely required.

## Root of the BPE problem

Open `node_modules/@huggingface/transformers/src/pipelines.js` and search inside `TokenClassificationPipeline.call`. Inside you'll see the literal comment:

```js
// TODO: Add support for start and end
```

That is, the pipeline **knows** these fields ought to be there, but it's not implemented yet. In v4.2 this TODO is still open. What we get:

```js
[
  { entity_group: 'PER', score: 0.998, word: 'John Smith' },
  { entity_group: 'LOC', score: 0.987, word: 'San Francisco' }
]
```

The decoded `word` is `tokenizer.decode(tokens, { skip_special_tokens: true })` over the grouped tokens. What's wrong with it:

1. **Whitespace around punctuation.** For input `John, Smith` the decoder may emit `John , Smith` — because in the BERT tokenizer comma is usually a separate token, and on decode it gets sewn back together with surrounding spaces.

2. **Case.** If we used an uncased model (`bert-base-uncased-ner`), the decoded `word` would come back lowercased: `john smith` for source `John Smith`. Our `Xenova/bert-base-NER` is cased, so this problem isn't acute in our code, but the code handles it via the regex flag `i`.

3. **Subword artifacts.** If a name is a subword (`Schwarzschild` → `Sch ##war ##zsch ##ild`), the decoder usually glues them without artifacts, but sometimes leaves strange spaces.

The solution — **searchably reconstruct the span in the source text** via linear scan: for each entity look up its `word` in `text`, starting from the cursor (protection against repeated names). That's `findEntityCharRange`.

## HOW (BPE → char)

`src/alignment/bpe-to-char.ts`:

```ts
import type { Span } from '../types/span';

export type CharRange = { start: number; end: number };

/**
 * Try to find `word` in `text` at-or-after `fromIndex`. Returns null if no match.
 * Tolerates extra spaces inside `word` (BERT decode often inserts a space around
 * punctuation: "John , Smith" for input "John, Smith").
 */
export function findEntityCharRange(
  text: string,
  word: string,
  fromIndex: number,
): CharRange | null {
  const trimmed = word.trim();
  if (trimmed.length === 0) return null;

  // 1. Fast path: literal substring search.
  const direct = text.indexOf(trimmed, fromIndex);
  if (direct !== -1) {
    return { start: direct, end: direct + trimmed.length };
  }

  // 2. Whitespace-tolerant search. Build a regex that allows any whitespace where the
  // decoded word has whitespace, and optional whitespace around punctuation.
  const escaped = trimmed
    .split(/(\s+)/)
    .map((part) => {
      if (/^\s+$/.test(part)) return '\\s+';
      // Allow optional whitespace between adjacent characters when one side is punctuation.
      return part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    })
    .join('');
  const re = new RegExp(escaped, 'i');
  const slice = text.slice(fromIndex);
  const m = slice.match(re);
  if (m && m.index !== undefined) {
    return { start: fromIndex + m.index, end: fromIndex + m.index + m[0].length };
  }

  return null;
}
```

Step-by-step.

### Fast path (`indexOf`)

In 90% of cases the decoded `word` matches the original substring literally. `indexOf(trimmed, fromIndex)` finds the substring in O(n). This is the hot path — for typical names ("John Smith", "Berlin") it works without any regex magic.

`fromIndex` is the `cursor` passed from `aggregateMlSpans`. It protects against two problems:

1. **Repeated names.** If "John" appears twice in the text and BERT returned two entities, the second should map to the second occurrence. `indexOf` with `fromIndex` guarantees this.

2. **Out-of-order matches.** Without `fromIndex` the second entity could accidentally match the first occurrence (if BERT returned them in an unnatural order). With `fromIndex` — guaranteed left-to-right.

### Whitespace-tolerant fallback

If literal search didn't find (typical case — BERT decoded `"John , Smith"` for source `"John, Smith"`), build a regex.

First `trimmed.split(/(\s+)/)`. This is a **split with capture group** — it doesn't discard the separators, it leaves them in the array. For `"John , Smith"` we get `["John", " ", ",", " ", "Smith"]` (simplified; in practice splitting on `\s+` merges adjacent spaces into one element).

Then mapping:
- Whitespace elements (`/^\s+$/`) become `\s+` (one or more of any whitespace in regex).
- Non-whitespace elements are escaped (`[.*+?^${}()|[\]\\]` — all regex meta-characters).

Result: `John\s+,\s+Smith` (regex pattern). Matches `"John, Smith"`, `"John , Smith"`, `"John\t,\nSmith"` — any whitespace variation.

The `i` flag — for case insensitivity (just in case of uncased models).

### Sanity check

If both paths missed — return `null`. `aggregateMlSpans` will drop such a span. Better to lose an entity than to create one with broken offsets.

Additionally there's `validateOffsets`, which asserts the invariant: `text.slice(s.start, s.end) === s.text`. In dev this catches alignment regressions; whether to run it in prod is debatable (overhead is minimal, but throwing in prod over a minor desync isn't great).

```ts
export function validateOffsets(spans: Span[], text: string): void {
  for (const s of spans) {
    if (s.start < 0 || s.end > text.length || s.start >= s.end) {
      throw new Error(
        `Span out of bounds: type=${s.type} ${s.start}..${s.end}, text.length=${text.length}`,
      );
    }
    const slice = text.slice(s.start, s.end);
    if (slice !== s.text) {
      throw new Error(
        `Span text mismatch: expected ${JSON.stringify(slice)}, got ${JSON.stringify(s.text)}`,
      );
    }
  }
}
```

## HOW (char → DOM Range)

Now we have `{ start: 27, end: 38 }` — character offsets in the plain-text projection of the editor content. But `<contenteditable>` has HTML structure:

```html
<div id="editor" contenteditable="true">
  Reach <strong>John Smith</strong> at 142 Maple St.
</div>
```

Plain-text view (concatenation of all text nodes in DOM order):
```
"Reach John Smith at 142 Maple St."
```

Span `{ start: 6, end: 16 }` is `"John Smith"`. But in the DOM that text sits **entirely inside the `<strong>` text node**, and the neighbor text nodes are `"Reach "` (before `<strong>`) and `" at 142 Maple St."` (after).

`charRangeToDomRange` solves this problem:

```ts
export function charRangeToDomRange(
  root: HTMLElement,
  start: number,
  end: number,
): Range {
  if (end < start) {
    throw new Error(`charRangeToDomRange: end (${end}) < start (${start})`);
  }
  const doc = root.ownerDocument!;
  const range = doc.createRange();

  const walker = doc.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    nodes.push(n as Text);
  }

  if (nodes.length === 0) {
    range.setStart(root, root.childNodes.length);
    range.setEnd(root, root.childNodes.length);
    return range;
  }

  // Compute prefix lengths so node `i` covers chars [offsets[i], offsets[i+1]).
  const offsets = new Array<number>(nodes.length + 1);
  offsets[0] = 0;
  for (let i = 0; i < nodes.length; i++) {
    offsets[i + 1] = offsets[i] + nodes[i].data.length;
  }
  const total = offsets[nodes.length];

  const clampedStart = Math.max(0, Math.min(start, total));
  const clampedEnd = Math.max(clampedStart, Math.min(end, total));

  // Pick the start node: smallest i such that clampedStart < offsets[i+1], else last.
  let startIdx = nodes.length - 1;
  for (let i = 0; i < nodes.length; i++) {
    if (clampedStart < offsets[i + 1]) {
      startIdx = i;
      break;
    }
    if (clampedStart === offsets[i + 1]) {
      // Boundary: try to find a non-empty next node.
      let j = i + 1;
      while (j < nodes.length && nodes[j].data.length === 0) j++;
      startIdx = j < nodes.length ? j : i;
      break;
    }
  }

  // Pick the end node: largest i such that clampedEnd > offsets[i], else first.
  let endIdx = 0;
  for (let i = nodes.length - 1; i >= 0; i--) {
    if (clampedEnd > offsets[i]) {
      endIdx = i;
      break;
    }
  }

  range.setStart(nodes[startIdx], clampedStart - offsets[startIdx]);
  range.setEnd(nodes[endIdx], clampedEnd - offsets[endIdx]);
  return range;
}
```

### TreeWalker

`doc.createTreeWalker(root, NodeFilter.SHOW_TEXT)` creates an iterator that traverses **only text nodes** in document order. It's robust to nested elements: text inside `<strong>` → text inside `<em>` inside `<div>` — all pass through in the right order.

We materialize the walker into an array `nodes: Text[]` — we'll need bidirectional iteration later.

### Prefix sums

```ts
offsets[0] = 0;
for (let i = 0; i < nodes.length; i++) {
  offsets[i + 1] = offsets[i] + nodes[i].data.length;
}
```

Standard trick: `offsets[i]` is the plain-text offset of the start of the i-th text node. After the loop `offsets[nodes.length]` = `total` plain-text length.

Invariant: text node `i` covers character range `[offsets[i], offsets[i+1])`.

### Finding startIdx

We want the **smallest i** such that `clampedStart < offsets[i+1]`. That is, the first text node that contains `clampedStart`. Linear search.

**Subtle spot — boundary handling.** When `clampedStart === offsets[i+1]` (the offset is exactly on the boundary between the i-th and (i+1)-th), where do we put it — in i (as exclusive end) or in i+1 (as inclusive start)? We want **in i+1**, so the Range "hugs" the selection rather than slipping to the end of the previous node (especially when the previous is in another `<strong>`).

Another subtlety: between two non-empty text nodes there can be an empty text node (e.g. after DOM normalization). We skip over them:

```ts
let j = i + 1;
while (j < nodes.length && nodes[j].data.length === 0) j++;
startIdx = j < nodes.length ? j : i;
```

### Finding endIdx

Mirror logic, but iterating from the back: **the largest i** such that `clampedEnd > offsets[i]`. That is, the last text node that starts before `clampedEnd`.

Without boundary skip — for end the simple logic suffices, because `range.setEnd` correctly handles offset == data.length (it points to the end of the node's content).

### Creating the Range

```ts
range.setStart(nodes[startIdx], clampedStart - offsets[startIdx]);
range.setEnd(nodes[endIdx], clampedEnd - offsets[endIdx]);
```

`clampedStart - offsets[startIdx]` is the offset inside the text node (local), not global. `range.setStart` expects local.

## Tests you should write

At minimum 6 cases for `charRangeToDomRange`:

1. **Whole text in one text node.** `<div>John Smith</div>`, span [0..10] → Range entirely inside one Text node.
2. **Text spans 2 text nodes.** `<div>John <strong>Smith</strong></div>`, span [0..10] (= "John Smith") → setStart in the first Text, setEnd inside the Strong-Text.
3. **Text contains a `<br>`.** `<div>line1<br>line2</div>`. `<br>` is an element node, not text. TreeWalker skips it, two text nodes — `"line1"` and `"line2"`.
4. **Text contains a nested `<span>`.** `<div>before<span>middle</span>after</div>` — three text nodes.
5. **Empty text node between two non-empty.** After `node.normalize()` they usually disappear, but via DOM mutations they may remain.
6. **End offset > total length.** Should clamp, not crash.

In `src/alignment/__tests__/char-to-dom.test.ts` all these cases should exist. They're your safety net against regressions — you can't add a new DOM structure without thinking about alignment.

## Failure modes

**Invisible characters (zero-width, RTL marks).** TreeWalker sees them as part of the text node's `data.length`, and they count in plain-text offsets. If BERT decoded without them, but the source text has them — `indexOf` will miss. Defense: `findEntityCharRange` falls back to a whitespace-tolerant regex, but zero-width characters aren't whitespace — a potential hole. In practice in business text they aren't there.

**Line break `\n` vs `<br>`.** In `<contenteditable>` Enter usually turns into `<div><br></div>` or `<p>...</p>`. Browser-specific: Chrome prefers `<div>`, Firefox — `<br>`. Plain-text projection via TreeWalker doesn't account for this: text node `"line1"` + `<br>` + text node `"line2"` glues into `"line1line2"` without a line break. For alignment to work on multiline text you need either to inject `\n` while walking, or to process `<br>` via NodeFilter.SHOW_ELEMENT. In the current code this is a simplification — multiline editing isn't first-class supported.

**Combining marks.** `é` as two codepoints (`e` + combining acute). JS `string.length` counts them separately, DOM Range too. BERT tokenizer usually normalizes via NFC. If the source text is in NFD and the decode in NFC — `indexOf` misses. Defense: you can normalize `text` via `text.normalize('NFC')` before regex pass and inference. Not in the current code — TODO for production.

**Emoji and surrogate pairs.** `'😀'.length === 2` in JS. `text.slice(0, 1)` gives a broken surrogate. If a span lands on half of a surrogate — the Range is valid, but renders garbage. Defense: check that start/end don't cut a surrogate pair. In our NER scenario PII usually doesn't contain emoji, but this is a failure mode for the general case.

**ML returned a completely different word.** If the decoded `word` differs radically from the source (e.g. BERT added extra tokens during aggregation), `findEntityCharRange` will return `null`, and the span will drop. Worse — if it accidentally found `word` somewhere else (e.g. "Berlin" appears twice, and the second occurrence mistakenly matched the first). The cursor pattern protects against this; but if the cursor drifted — ghosts can appear. Tests on repeated names are mandatory.

## Performance

`findEntityCharRange` is O(n) per entity, total O(n × k) for k entities in text of length n. On a typical phrase (200 chars, 5 entities) — 0.1 ms. Negligible.

`charRangeToDomRange` is O(m + k) for m text nodes and k boundary checks. For an editor with 10–100 text nodes — <1 ms. If the editor were CodeMirror-style with thousands of nodes per syntax token, you could build a prefix-sums cache and reuse it between spans. We have a simple `<div contenteditable>` — no need to optimize.

## What should click

1. **Why BERT doesn't give offsets** — because of the library TODO, you have to reconstruct by search.
2. **Fast path + fallback** — `indexOf` covers 90%, regex with `\s+` covers the edge case with punctuation.
3. **Cursor pattern in `aggregateMlSpans`** — guarantees repeated names don't collapse.
4. **TreeWalker → prefix sums → setStart/setEnd** — standard technique for char → DOM Range mapping.
5. **Boundary tie-break** — at the junction of two text nodes we prefer start → next, end → previous, so the Range "hugs" the entity.
6. **`validateOffsets`** — sanity check in dev; optional in prod.

Next chapter — how to make this whole machinery keep up with a fast typist.
