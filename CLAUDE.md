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

### Hybrid Scoring
```
0.5 * bm25_normalized + 0.5 * (cosine_sim + 1.0) / 2.0
```
- Combines lexical (BM25) + semantic (cosine similarity) scores
- Both normalized to [0, 1]
- Reranks BM25 results with vector similarity

### Lazy Initialization
- Embedder initialized via `OnceLock<Result<Embedder, String>>`
- Model loaded on first query, not at startup
- Stored in singleton for reuse across requests
