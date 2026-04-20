# rs-search — Rust CLI/MCP Server for Codebase Search

Hybrid BM25 + vector semantic search for codebases. Uses candle-core ML framework with nomic-embed-text-v1.5 embeddings.

## Key Technical Constraints

### Nightly Rust Required
candle-core 0.10.2 uses `usize::is_multiple_of()` (tracking issue #128101), an unstable feature gated under `unsigned_is_multiple_of`. This is rejected by stable Rust (E0658) even on 1.94.1. The project requires nightly toolchain.

**Configuration:**
- `rust-toolchain.toml`: `channel = "nightly"`
- CI: `dtolnay/rust-toolchain@nightly`

### Windows: MSVC cl.exe Required
`candle-core` → `tokenizers[onig]` → oniguruma C library (cc-rs build). The oniguruma library builds with MSVC, not MinGW. Windows local development requires MSVC cl.exe in PATH (Visual Studio Build Tools with C++ workload). GitHub Actions `windows-latest` has MSVC pre-installed.

## PDF Ingestion

`.pdf` files are first-class search targets. The scanner (`src/scanner.rs`) dispatches by extension to `src/pdf.rs::pdf_chunks`, which runs `pdf_extract::extract_text_from_mem` and splits on form-feed (`\x0c`) into one Chunk per page. Each page Chunk carries `line_start = line_end = page_number` — search hits render as `doc.pdf:<page>`.

- **Crate**: `pdf-extract = "0.9"` — pure Rust, no C deps, builds under MSVC nightly alongside oniguruma.
- **Cache**: extracted pages persist to `.code-search/pdf-cache/<hash>.json` keyed on `abs_path + mtime`; rescans skip extraction.
- **Ignore filter**: `.pdf` is in `CODE_EXTENSIONS` and removed from `BINARY_EXTENSIONS` in `src/ignore.rs`. Both changes are required — `should_ignore` rejects anything not in `CODE_EXTENSIONS` *or* present in `BINARY_EXTENSIONS`.
- **Limits**: honors the existing 50MB cap. Encrypted, scanned-only, or malformed PDFs yield zero chunks silently (no OCR). Digital PDFs extract fully.

## Vector Search Architecture

### Embedding Model
- **File**: nomic-embed-text-v1.5.Q4_K_M.gguf (split into 6 parts for size)
- **Loading**: `gguf_file::Content::read` → dequantize tensors → `VarBuilder::from_tensors` → `NomicBertModel`
- **Computation**: mean-pooling + L2 normalization
- **Code**: `src/embed.rs`

### Tokenization
- **File**: `models/tokenizer.json` (711KB, bundled via `include_bytes!`)
- **Type**: BERT WordPiece (not BPE)
- **Why separate**: nomic-embed-text-v1.5 requires WordPiece; GGUF-embedded tokenizer only supports BPE

### Hybrid Scoring — Reciprocal Rank Fusion
Fusion lives in `src/fusion.rs`. RRF with `k=60` merges BM25 and vector rankings; BM25 gets a `1.5×` weight when `looks_like_identifier(query)` is true (snake_case, kebab-case, dotted, or camelCase without spaces). The final score is normalized to `[0, 1]`.

### Lazy Initialization
- Embedder initialized via `OnceLock<Result<Embedder, String>>`
- Model loaded on first query, not at startup
- Stored in singleton for reuse across requests

### SIMD Vector Ops
`embed::cosine` uses `simsimd::SpatialSimilarity::cosine` when SSE is available, with a scalar fallback. Expect 5–20× speedup over hand-rolled scalar math on AVX2/AVX-512/NEON hosts.

### Content-Addressable Embedding Cache
`src/embed_cache.rs` keys BLAKE3(`model_tag || dim || text`) → `f32` vector, in-memory (Mutex<HashMap>) + on-disk at `.code-search/emb-cache/<hex>.bin`. Skips re-embedding unchanged chunks across runs. Dim is part of the key so MRL truncation produces a separate cache lane.

### Matryoshka Truncation
Set `RS_SEARCH_DIM=256` (or any value less than full dim) and the embedder slices the vector and renormalizes. No retraining needed — nomic-embed-text-v1.5 is MRL-trained.

### Embedder Prompt Prefixes
Defaults are nomic's `search_query: ` / `search_document: `. Override via `RS_SEARCH_QUERY_PREFIX` and `RS_SEARCH_DOC_PREFIX` for CodeRankEmbed (`Represent this query for searching relevant code: `) or other embedders.

## Feature Gates

- `default = ["vector", "perf"]`
- `vector` — gates `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`. Disabling shrinks the binary to pure-Rust BM25 + RRF + PDF scan.
- `perf` — gates `mimalloc` as `#[global_allocator]`. Turn off for musl-only builds or when another allocator is preferred.

## Subcommands

- `rs-search <query...>` — one-shot search (legacy positional).
- `rs-search search <query...>` — explicit subcommand.
- `rs-search serve` — MCP stdio server (also the default when no args given).
- `rs-search explain <query...>` — per-token IDF, doc frequency, RRF weights, matched tokens per result (`src/explain.rs`).

## MCP Panic Boundary

`tools/call` is wrapped in `std::panic::catch_unwind`. A handler panic emits a JSON-RPC `-32603` error instead of killing the session. Query input is capped via `inputSchema.properties.query.maxLength=8192`.

## Scanner

`src/scanner.rs` uses `ignore::WalkBuilder` (ripgrep/fd's crate) with `.gitignore`, `.git/info/exclude`, global gitignore, and `.codesearchignore` all respected out of the box. The custom `IGNORED_DIRS` list still guards against vendored caches even when no ignore file exists.

## Eval Harness

`src/eval.rs` exposes `ndcg_at_k`, `mrr`, `recall_at_k`, `precision_at_k` plus an `EvalReport` aggregator. Plug in BEIR/CoIR qrels to gate NDCG@10 regressions in CI.

## Release Tooling

- `release-plz.toml` — changelog + version-bump PRs driven by conventional commits.
- `[workspace.metadata.dist]` in `Cargo.toml` — cargo-dist cross-platform artifacts.
- `.github/workflows/release-plz.yml` — opens release PRs on pushes to `main`.
