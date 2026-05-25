# Fixtures

`synthetic.json` is the gold corpus for `eval/harness.ts` and `eval/benchmark.ts`. Each
record is `{ id, text, expectedSpans: [{ type, start, end }] }`. Spans use
character offsets in the unicode-decoded text — same convention as
`Span.start`/`Span.end` produced at runtime.

Phase 10 commits 10 baseline fixtures across the 6 categories. A human is asked to add
20 more per ROADMAP Phase 11. Mix to aim for:

- Clinical (target 6 total)
- Legal (target 6 total)
- Financial (target 6 total)
- Conversational (target 4 total)
- Mixed (target 4 total)
- Edge cases (target 4 total)

Tip: the easiest authoring loop is to write the text first, run the regex extractors
to find offsets for EMAIL/PHONE/SSN/DATE/ADDRESS, then add PERSON/LOCATION expected
spans manually. Verify offsets by ensuring `text.slice(start, end)` equals the entity
substring.
