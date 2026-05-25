# Hybrid pipeline — regex + ML and four merge rules

## WHY

The temptation of "one big model for everything" is strong but poorly grounded. Think about EMAIL:

```
john.smith@example.com
```

What will find it better — a fine-tuned BERT NER or the regex `/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/`?

Regex gives ~99% precision and ~99% recall — the email format is effectively machine-checkable (with minimal stretches: TLD length 2+, permitted characters in local-part, etc.). BERT NER can find it, but with worse precision on boundaries (inside "Email: john@x.com please reply" the model often eats the colon or "please" into the span), and it costs 45 ms inference vs 0.1 ms regex execution.

Same for SSN (`\b\d{3}-\d{2}-\d{4}\b`), PHONE (`(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}`), DATE (ISO + named months), ADDRESS (number + street + suffix).

And vice versa: try finding a person via regex in "I talked to Anton Petrov on Tuesday." What pattern? "I talked to" — an anaphoric phrase, it isn't always written that way. Names come in two-part, three-part, with initials. Without context — no way. BERT trained on a NER dataset catches this naturally.

**Hybrid pipeline** = regex for structured patterns + BERT for unstructured. Each tool in its mode. This isn't a compromise, it's a proper separation of responsibility.

The price of hybrid is the need to **combine two span streams** into one list, without overlaps. That's the merge layer, the main locus of complexity in this chapter.

## HOW (regex side)

`src/pipeline/regex.ts` — five extractors, each returns `Span[]`:

```ts
import type { EntityType, Span } from '../types/span';

function makeSpan(type: EntityType, start: number, end: number, text: string): Span {
  return { type, start, end, text, source: 'regex', confidence: 1.0 };
}

function runRegex(text: string, type: EntityType, re: RegExp): Span[] {
  const out: Span[] = [];
  for (const m of text.matchAll(re)) {
    if (m.index === undefined) continue;
    out.push(makeSpan(type, m.index, m.index + m[0].length, m[0]));
  }
  return out;
}

export const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;
export function extractEmails(text: string): Span[] {
  return runRegex(text, 'EMAIL', EMAIL_RE);
}

// North-American phone numbers, optional country code, optional parens around area code.
export const PHONE_RE =
  /(?<![\d-])(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)/g;

export const SSN_RE = /\b\d{3}-\d{2}-\d{4}\b/g;

const MONTH =
  '(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)';
export const DATE_ISO_RE = /\b\d{4}-\d{2}-\d{2}\b/g;
export const DATE_NAMED_RE = new RegExp(`\\b${MONTH}\\s+\\d{1,2}(?:,\\s*\\d{2,4})?\\b`, 'g');

export const ADDRESS_RE =
  /\b\d{1,6}\s+[A-Za-z][A-Za-z0-9.\s]*?\s(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Way|Drive|Dr|Court|Ct)\b\.?/g;
```

Notable nuances:

**All regexes with the `/g` flag** — otherwise `matchAll` won't work. Without `/g` it throws TypeError "matchAll must be called with a global RegExp".

**Lookbehind/lookahead in PHONE_RE** — `(?<![\d-])` prevents the phone from matching inside a long digit sequence (e.g. in `SSN 555-12-3456-789` — the chunk `555-12-3456` should not land in PHONE), `(?!\d)` — same on the right. Lookbehind is supported in all major browsers today.

**`confidence: 1.0`** for all regex spans — a semantic marker that these spans are "structured truth." In the merge layer regex wins over ML on partial overlap precisely because of this.

**`source: 'regex'`** — discriminant for merge. Without it merge can't tell an ML span from a regex span on identical extent.

Running all:

```ts
export function extractAllRegex(text: string): Span[] {
  return [
    ...extractEmails(text),
    ...extractPhones(text),
    ...extractSSN(text),
    ...extractDates(text),
    ...extractAddresses(text),
  ].sort((a, b) => a.start - b.start);
}
```

Sorting by `start` is a precondition for the merge layer (it works correctly on sorted input).

## HOW (ML side)

In the Worker after `ner()` we get raw Transformers.js output. With `aggregation_strategy: 'simple'` it's an array of objects like:

```js
[
  { entity_group: 'PER', score: 0.998, word: 'John Smith', start: undefined, end: undefined },
  { entity_group: 'LOC', score: 0.987, word: 'San Francisco', start: undefined, end: undefined },
  // ...
]
```

`start`/`end` — `undefined`. This is the known TODO in Transformers.js v4 (see chapter 4 in detail). So `aggregate.ts` reconstructs the offsets itself:

```ts
// src/worker/aggregate.ts (fragment)
function tagToEntityType(tag: string): EntityType | null {
  switch (tag) {
    case 'PER':
      return 'PERSON';
    case 'LOC':
      return 'LOCATION';
    default:
      return null; // ORG / MISC are intentionally dropped
  }
}

export function aggregateMlSpans(text: string, raw: unknown): MlSpan[] {
  if (!Array.isArray(raw)) return [];
  const spans: MlSpan[] = [];
  let cursor = 0;

  for (const r of raw as RawAggregated[]) {
    const tag = r.entity_group;
    if (!tag) continue;
    const word = r.word;
    if (!word) continue;
    const type = tagToEntityType(tag);
    if (!type) continue;

    const range = findEntityCharRange(text, word, cursor);
    if (!range) continue;
    cursor = range.end;

    spans.push({
      type,
      start: range.start,
      end: range.end,
      text: text.slice(range.start, range.end),
      source: 'ml',
      confidence: r.score ?? 0,
    });
  }
  return spans;
}
```

Key points:

**`tagToEntityType`** does filtering: only `PER` → `PERSON` and `LOC` → `LOCATION`. `ORG` and `MISC` — intentionally dropped (see chapter 1, explanation). This gives +precision at the cost of recall, but in the context of PII redaction precision is more important.

**Cursor pattern.** `findEntityCharRange(text, word, cursor)` searches for `word` in `text`, starting from `cursor`. After every found span — `cursor = range.end`. This protects against the case where the same name occurs twice and both times lands in the NER output: the first matches the first occurrence, the second matches the second, because `indexOf` starts from `cursor`.

**Skipping unparseable spans.** If `findEntityCharRange` returned `null` (the decoded `word` is not found in the source text), the span is dropped. Better to lose a span than to emit it with wrong offsets — the latter breaks DOM Range alignment downstream.

**`source: 'ml'`** + **confidence from the model** — input for merge.

Details of `findEntityCharRange` — in chapter 4.

## HOW (merge layer)

Regexes and BERT can return overlapping spans. Example (contrived, but typical):

```
Text: "Reach John Smith at 142 Maple St or jsmith@x.com by March 14, 2026."

regex EMAIL:   "jsmith@x.com" [37..49]
regex ADDRESS: "142 Maple St" [20..32]
regex DATE:    "March 14, 2026" [53..67]
ML PERSON:     "John Smith" [6..16]
ML LOCATION:   "Maple St" [24..32]   ← overlaps with regex ADDRESS [20..32]
```

ML found "Maple St" as LOCATION, regex found "142 Maple St" as ADDRESS. This is **partial overlap with containment**: ADDRESS ⊃ LOCATION. What to do?

`src/pipeline/merge.ts` solves this via four rules:

```ts
// Merge regex spans (high precision, structured) with ML spans (PERSON/LOCATION) into
// a single non-overlapping span list per the rules in ARCHITECTURE.md:
//
//   1. Full containment: if span A fully contains span B, keep the larger A and drop B.
//   2. Partial overlap: structured (regex) wins over ML.
//   3. Equal extent: regex wins on the type tag, but kept ML confidence drops.
//      (Implementation: always prefer regex on equal extent.)
//   4. Disjoint: keep both.
```

Implementation:

```ts
export function mergeSpans(regexSpans: Span[], mlSpans: Span[]): Span[] {
  const all = [...regexSpans, ...mlSpans].sort((a, b) => a.start - b.start);
  const kept: Span[] = [];

  for (const candidate of all) {
    let drop = false;
    for (let i = 0; i < kept.length; i++) {
      const existing = kept[i];
      const rel = relate(existing, candidate);
      if (rel === 'disjoint') continue;
      // We have an interaction — resolve it.
      const winner = pickWinner(existing, candidate, rel);
      if (winner === existing) {
        drop = true;
        break;
      } else {
        // Replace the existing kept span with candidate.
        kept.splice(i, 1);
        i--; // re-check remaining kept (rare cascading overlap)
      }
    }
    if (!drop) kept.push(candidate);
  }
  return kept.sort((a, b) => a.start - b.start);
}
```

And helper functions:

```ts
type Relation = 'disjoint' | 'a-contains-b' | 'b-contains-a' | 'equal' | 'partial';

function relate(a: Span, b: Span): Relation {
  if (a.end <= b.start || b.end <= a.start) return 'disjoint';
  if (a.start === b.start && a.end === b.end) return 'equal';
  if (a.start <= b.start && a.end >= b.end) return 'a-contains-b';
  if (b.start <= a.start && b.end >= a.end) return 'b-contains-a';
  return 'partial';
}

function pickWinner(a: Span, b: Span, rel: Relation): Span {
  if (rel === 'equal') {
    if (a.source === 'regex' && b.source !== 'regex') return a;
    if (b.source === 'regex' && a.source !== 'regex') return b;
    return a; // tie — keep first seen
  }
  if (rel === 'a-contains-b') return a;
  if (rel === 'b-contains-a') return b;
  // partial:
  if (a.source === 'regex' && b.source !== 'regex') return a;
  if (b.source === 'regex' && a.source !== 'regex') return b;
  return (a.end - a.start) >= (b.end - b.start) ? a : b;
}
```

Breaking down by rule:

### Rule 1 — full containment

A fully contains B → keep A, drop B. Logic: if ML found "John Smith" [6..16], and separately regex found "John" [6..10] (hypothetically, not in our code, but imagine) — keep "John Smith". We prefer fuller names.

In our example: ADDRESS [20..32] ⊃ LOCATION [24..32] → keep ADDRESS, drop LOCATION. This is right: the user probably wants to redact the whole address string, not just the street name.

### Rule 2 — partial overlap

A and B overlap, but neither contains the other → regex beats ML. Example: regex PHONE "(555) 123-4567" [10..24] and ML LOCATION "California 555" [0..14] (synthetic). They partially overlap. Regex wins because its precision is higher on a structured pattern.

### Rule 3 — equal extent

A and B have exactly the same boundaries (start and end match) → regex wins on the tag. Example: ML LOCATION "March 14" [40..48] (model erred) and regex DATE "March 14" [40..48]. Keep DATE. Realistically — yes: BERT NER sometimes tags dates as LOCATION (especially "March" by itself), regex is more authoritative here.

### Rule 4 — disjoint

A and B don't overlap → keep both. Trivial.

### Cascading replacement

A subtle point in imperative `mergeSpans`: when a candidate wins over an existing, we do `splice + i--`. This is needed so that after removing `existing` the loop re-checks all other kept spans for conflict with the new candidate. In practice cascading happens rarely, but if we didn't do this — a pair of overlapping spans could remain.

## Composition: where merge runs

In this project merge runs **in the main thread**, not in the worker, and **after** the worker's response:

```
1. text → main thread takes a snapshot, increments version
2. main thread → postMessage({ type: 'infer', text, version })
3. in parallel: main thread → extractAllRegex(text) → regexSpans[]
4. Worker → inference + aggregate → MlSpan[]
5. Worker → postMessage({ type: 'inferred', spans, version, latencyMs })
6. main thread: version check → if fresh, mergeSpans(regexSpans, mlSpans) → finalSpans[]
7. main thread: charRangeToDomRange + render badges
```

Why regex in main thread, not in worker? Regex on 200 chars is <0.5 ms, no problem for the main thread. Passing text twice (once to worker, then back) is more expensive than computing in place. Plus, merge fundamentally needs to know both lists at once, and the main thread is the natural place for them to meet (it holds the snapshot text and version).

## Failure modes

**Invalid spans from regex.** If regex accidentally matches "across" — the span will have type EMAIL, and the user will see "across" redacted. We have `validateOffsets` in the alignment module for this case, which throws in dev if `text.slice(start, end) !== span.text`. In prod this shouldn't happen, but a sanity check is useful.

**Merge left overlapping spans.** Hypothesis: if there are three spans with triple overlap, can cascading replacement miss a case? We have unit tests in `pipeline/__tests__/merge.test.ts` covering triple overlaps. If they slip out — DOM Range alignment may crash on overlapping highlights.

**ML output in an unexpected format.** Transformers.js could in a minor release change the structure of the returned object (e.g. `entity_group` → `entityGroup` in camelCase). In v4 with `aggregation_strategy: 'simple'` the field is always `entity_group`, and `aggregate.ts` reads it directly. At the next major upgrade you should run our eval harness against fixtures — if F1 dropped sharply, it's the shape of the output.

**Cursor pattern missed.** If BERT returns entities in an unexpected order (not left-to-right by appearance in text), the cursor pattern can miss. In practice with `aggregation_strategy: 'simple'` entities come left-to-right. Tests cover this invariant.

**Regex performance on huge texts.** If the user pasted 100 KB of text, `extractAllRegex` may take >100 ms. We have debounce 200 ms which smooths this, but on giant pastes you should additionally cap snapshot size (e.g. process only the visible viewport part). In the current code there's no such cap — a project simplification.

## What should click

1. **Hybrid is not a compromise, it's a proper separation.** Structured patterns → regex; unstructured → ML.
2. **Merge is explicit, testable logic.** Four rules, not "merge somehow."
3. **Source field is the discriminant.** Regex vs ML — fundamentally different trust levels, and they differ in the span's type.
4. **Cursor pattern in aggregate** — guarantees repeated names don't collapse into one span.

Next chapter — the hardest: how BERT output without offsets becomes a correct char-offset → DOM Range. Without this nothing will highlight.
