# Decisions

## 1. Hybrid regex + ML over ML-only

I chose a hybrid pipeline (regex for EMAIL/PHONE/SSN/DATE/ADDRESS, BERT-NER for PERSON/LOCATION) over a purely-ML approach because:

- The structured categories have a regex precision near 1.0; an ML model on them gives boundary-noisy spans that need post-processing anyway. EMAIL is the worst offender — BERT happily breaks `john.smith@example.com` into 3 tokens with mismatched B-/I- tags.
- The unstructured categories (names, places) have no fixed pattern, so ML earns its keep there.
- Sending only ~2 categories to the model also reduces the merge-conflict surface.

The cost is a non-trivial merge layer (`pipeline/merge.ts`) with four overlap rules: full containment / partial overlap / equal extent / disjoint. Each rule is unit-tested in `pipeline/__tests__/merge.test.ts`. If we ever add a 7th category, the overlap rules need to be re-thought.

## 2. `Xenova/bert-base-NER` (110 MB) over `distilbert-base-uncased-finetuned-conll03-english` (65 MB)

I chose the larger BERT-base over distilbert because:

- Distilbert is uncased — for PERSON entities in narrative text, that costs measurable F1 (the lowercasing eats the cue that "Adams" is a name and not the verb "adams").
- 110 MB is downloaded once and cached by the browser. Subsequent loads are ~50 ms cold-start; the 45 MB delta only matters for first-run UX, which we already hide behind a labeled progress bar.
- bert-base-NER ships with verified ONNX weights and has been confirmed to run on WebGPU; the smaller distilbert variant has occasional WebGPU quirks (graph rewrites for fp16) that aren't worth debugging for this project.

The model is loaded through `@huggingface/transformers` v4 with `aggregation_strategy: 'simple'`, which groups B-/I- BPE tokens into word-level entities but does not populate `start`/`end` (upstream TODO). Character offsets are recovered in `src/alignment/bpe-to-char.ts` by linear cursor-following substring search with a whitespace-tolerant fallback — concentrating the alignment subtleties (the project's stated central challenge) in one unit-tested module (`alignment/__tests__/char-to-dom.test.ts`, 6 cases).

The cost is a slower first-load on slow networks (30–60 s on 4G). UX mitigates by labelling "Downloading model (one-time)" prominently; subsequent loads are instant.

## 3. Browser Cache Storage over plain IndexedDB

I chose to enable `env.useBrowserCache = true` and let Transformers.js v4 manage the cache, rather than wiring a custom OPFS layer.

- v4 uses Cache Storage (`caches.open`) under the hood, which is file-system-backed in Chromium and Firefox and behaves equivalently to OPFS for read latency on 100 MB blobs.
- Custom OPFS handling would require intercepting `fetch` inside the worker — manageable but adds a maintenance burden every time Transformers.js bumps its model loader.
- For a read-mostly workload like model weights, OPFS-specific niceties (truncation, append writes) buy nothing.

`src/storage/opfs-cache.ts` sets `env.useBrowserCache`, `env.allowRemoteModels`, and `env.allowLocalModels`. The file name is historical (OPFS was the planning-stage candidate); the implementation uses Cache Storage. A TODO note in that file flags the planned migration to the upcoming `ModelRegistry`-based config API in a future Transformers.js release.
