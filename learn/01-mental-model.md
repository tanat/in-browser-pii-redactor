# Mental Model — privacy as a system property

## WHY

In projects 02–04 you got used to one phrase in the privacy policy: "we don't retain your data after processing the request." That's a promise. Impossible to verify from the outside — you have to trust the product, the model provider, the infrastructure underneath them. Every link in the chain can slip: a developer accidentally logs the payload, the provider changes ToS, an SSN lands in a DC log and doesn't roll off retention in time.

This project is an attempt to build the privacy guarantee not as a promise but as **a system property derivable from the architecture**. If there's no fetch to a server for processing text in the code — data physically cannot leave. You can verify this in thirty seconds: open DevTools, open the Network tab, type sensitive text, look — empty.

Today this approach has stopped being exotic. Local-first AI has become a normal architectural option: on a high-end laptop a quantized 3B model responds faster than a round-trip to the OpenAI API on a single-turn request. Regulation (HIPAA, GDPR, AI Act) has turned "data doesn't leave the device" into a formal compliance argument, not just a marketing phrase.

## HOW (mentally)

Compare two pictures.

### Picture A — cloud pipeline (projects 02–04)

```
User keystroke
   ↓
React state
   ↓
fetch('/api/redact', { body: { text } })          ← data left the perimeter
   ↓
Vercel edge / route handler
   ↓
fetch('https://api.openai.com/v1/...', { body })  ← and again
   ↓
   ... model in the provider's data center, unknown retention
   ↓
JSON response → React → re-render
```

All privacy promises here are **soft promises** (logs, ToS, agreements). At least three places can leak: your Vercel logger, the provider's logger, network capture on an intermediate node.

### Picture B — local-first pipeline (this project)

```
User keystroke
   ↓
React state
   ↓
postMessage({ text, version }) → Web Worker
   ↓
worker scope: Transformers.js → ONNX Runtime Web → WebGPU
   ↓
inference on your GPU
   ↓
postMessage({ spans, version }) ← back to main thread
   ↓
React → re-render
```

The only `fetch` that runs on the page is the initial download of model weights from the Hugging Face CDN. After that the weights sit in Cache Storage (~110 MB) and the page works offline. You can physically disconnect the internet, and the editor will keep redacting PII.

This is **privacy as a system property**: the guarantee follows from the fact that there is no code in the build artifact that would send user text outward. You don't need to trust the developer — you can open the bundled JS and check.

## What "in the browser" means today

This stopped being a hack. Three technological bricks fell into place:

**WebGPU** — browser API for accessing GPU compute shaders. The current state of things:
- Chrome / Edge — long since (2023+), all platforms
- Safari 26 — on by default on macOS Tahoe 26, iOS 26, iPadOS 26, visionOS 26
- Firefox 141 — Windows (July 2025), Firefox 145 — macOS ARM64
- Firefox on Linux and Android — promised in 2026

For the project that means: on ~80% of users the pipeline runs on the GPU, on the rest — WASM fallback (~5× slower, but still works).

**WebAssembly (WASM) + SIMD + threads** — for devices without WebGPU. ONNX Runtime Web compiles kernels to WASM, uses SIMD instructions and shared memory for parallelism. On INT8 BERT-class models it gives ~150–300ms on a short phrase.

**Transformers.js v4** (February 2026) — JS wrapper over ONNX Runtime Web with a pipeline API in the spirit of the Python `transformers` library. In v4 the runtime was rewritten in C++, gained up to 4× speedup, bundle shrunk by 53%. Supports Mamba, MoE, MLA — i.e. state-of-the-art architectures, not just classical transformers.

**Cache Storage** — browser API for long-term storage of HTTP responses. The Chrome team officially recommends it as primary storage for AI models (for weights over ~500 MB they suggest OPFS, we have 110 MB — Cache Storage). Transformers.js uses it via `env.useBrowserCache = true`. Downloaded once — afterwards "instant."

## What "PII" means in this project

Seven categories:

| Category | Source | Why this way |
|-----------|----------|------------|
| EMAIL | Regex | Structured pattern, regex gives ~99% precision |
| PHONE | Regex | NA format with optional country/parens |
| SSN | Regex | Strict format `NNN-NN-NNNN` |
| DATE | Regex | ISO + named months ("March 14, 2026") |
| ADDRESS | Regex | Number + street + suffix (St/Ave/Rd/...) |
| PERSON | ML (BERT) | No regular pattern, context needed |
| LOCATION | ML (BERT) | Same — cities, countries, landmarks |

`ORG` and `MISC` from `bert-base-NER` are intentionally dropped (in `aggregate.ts` the line `return null; // ORG / MISC are intentionally dropped`). They produce many false positives on short snippets and overlap with `LOCATION` in cases like "Maple Street Capital."

That's the hybrid approach: the right tool for the right job. Regex is precise on structured patterns, ML is necessary on unstructured. Then a merge layer combines them into one list of spans.

## Anatomy of the privacy guarantee

For the guarantee to hold, you need three coordinated decisions:

**1. No server-side code in the build artifact.**

Vite builds an SPA. No API routes, no `getServerSideProps`, no server actions. Vercel deploy — static. If there were a server, a developer could accidentally add logging there, and the privacy guarantee would depend on their discipline.

**2. ML runs in a Web Worker, not on a server.**

A Worker is a different execution context of the same origin. It can issue fetches, but in this project — it doesn't. You can technically verify: open `inference.worker.ts` and confirm that there's only `postMessage`, `@huggingface/transformers`, and Cache Storage initialization there.

**3. Model weights are the only outbound fetch, and it goes to a trusted CDN.**

On first launch Transformers.js does `fetch('https://huggingface.co/Xenova/bert-base-NER/resolve/main/onnx/model.onnx')` (and tokenizer files). These are **public assets**, not user data. After that — Cache Storage hit. You can additionally verify SRI/checksum if you're paranoid.

## Failure modes

What can break if you treat this architecture carelessly:

**Accidental fetch with user data.** Someone added analytics "for PII-category distribution metrics" and hardcoded the text itself in there. The privacy guarantee fell apart. Defense — code review focused on network calls, no `fetch(... user_text ...)` in code.

**Logging in a Service Worker.** A Service Worker can intercept fetches and send them to analytics. In this project there are no Service Workers, and that's a deliberate decision.

**Third-party scripts.** Any analytics (Sentry, GA) is a potential leak channel. Solution — don't include them, or include them with explicit sample blocking, or use self-hosted and provably privacy-friendly.

**Browser extensions.** Extensions with "read all data on websites" permission see the DOM. That's outside our control — but we can at least not pretend we're protecting against them. In a demo, honestly say: "this protects against network-level leaks, but not malicious extensions."

**Someone else's WebGPU shader compiler bug.** ONNX Runtime Web compiles shaders — theoretically a shader could leak data through a timing side channel. Realistically — no, but academically such attack surface exists. For nation-state threats this matters; for typical corporate secrets — no.

## Comparison with alternatives

| Approach | Privacy | Quality | Latency | Complexity |
|--------|---------|----------|---------|-----------|
| GPT-class API with redaction prompt | Soft promise | Good | ~500–2000 ms | Low |
| Self-hosted LLM in customer DC | Control over logs | Good | ~100–500 ms | High |
| **In-browser BERT NER** (this project) | **System property** | Sufficient | **~45 ms** | Medium |
| Regex-only | System property | Bad on PERSON | <1 ms | Low |
| Client-side LLM (3B param, WebLLM) | System property | Very good | 100–500 ms + 1+ GB weights | High |

The strength of the BERT NER approach is a point on the Pareto curve "good enough for PII at minimal weights and low latency." 110 MB and 45 ms is a comfortable range. WebLLM with a 3B model will give better results on complex texts, but a 1+ GB load and noticeable latency make it unsuitable for live-typing UX.

In 2026 an interesting alternative appeared — **GLiNER** (Generalist and Lightweight NER, BERT-like encoder). Small/medium versions are licensed Apache 2.0, support zero-shot ("find me entities of category `medical_record_number`" without dedicated training). If the project were being built from scratch today, you'd want to at least compare GLiNER-small (~200 MB, but more flexible) with `bert-base-NER` (110 MB, fixed 4 classes). We stay on BERT-base-NER because for the typical 4 PII categories it's reliably certified and has a verified ONNX build.

## What should click after this chapter

If you write "redaction-as-a-service" the usual way — it's two API calls and a contract with a provider. Privacy — is a promise.

If you write it as an in-browser pipeline — it's:
- no server in the build artifact,
- ML inside a Web Worker using WebGPU/WASM,
- model weights as a public asset from a CDN, cached in Cache Storage,
- all user text in browser memory, never on the network.

And the main point — this can be **shown**, not explained. The Network tab in DevTools is the privacy guarantee demo. After this chapter you should clearly see the difference between the two approaches and understand in which scenarios the second is justified.

In the next chapters we'll break down what this approach looks like in code: how ML is isolated (Worker), how regex and BERT are combined (hybrid pipeline), how BERT output becomes a DOM Range (alignment), how the pipeline stays responsive to fast input (snapshot versioning), and how to steer all of it through Cache Storage / WebGPU fallback (model loading).
