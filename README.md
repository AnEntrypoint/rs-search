# rs-search

Hybrid **BM25 + vector** code search for codebases, as a CLI and an **MCP** server. Pure-Rust scanner with `.gitignore` discipline, PDF ingestion, content-addressable embedding cache, and Reciprocal Rank Fusion.

- BM25 lexical scoring with identifier-aware tokenization (snake / kebab / camel splits)
- Vector reranking via [`nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (GGUF, loaded with `candle-core`)
- RRF fusion (`k=60`) with a `1.5×` BM25 boost for identifier-shaped queries
- `.pdf` is a first-class search target — pages render as `doc.pdf:<page>`
- MCP stdio server with per-call `project_root` / `index` / `discipline` resolution
- Git-commit search: top-N commits ranked against the same query, BM25 + vector

---

## Installation

`rs-search` requires the **nightly** Rust toolchain (`rust-toolchain.toml` pins it) and, on Windows, **MSVC** `cl.exe` for `onig_sys`. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the full Windows env recipe.

```bash
cargo +nightly install --git https://github.com/AnEntrypoint/rs-search rs-search
```

Or build from source:

```bash
git clone https://github.com/AnEntrypoint/rs-search.git
cd rs-search
cargo build --release
```

Pre-built artifacts (`x86_64`/`aarch64` for Linux / macOS / Windows-MSVC) ship via `cargo-dist` on tagged releases.

---

## CLI

```text
rs-search [--root DIR] [--index DIR] [--discipline NAME] <subcommand>

Subcommands:
  search <query...>     Run a one-shot hybrid search and print BM25 + vector results
  explain <query...>    Per-token IDF, doc-freq, RRF weights, matched tokens per hit
  serve                 Run the MCP stdio server (also the default when no args given)

Flags:
  --root DIR            Project root (defaults to cwd)
  --index DIR           Explicit index directory; wins over --discipline
  --discipline NAME     Resolves to <root>/.gm/disciplines/<name>/code-search
  --features            Print enabled build features and exit
```

Examples:

```bash
# One-shot search in the current dir
rs-search search "reciprocal rank fusion"

# Explicit root and index dir
rs-search --root ./repo --index ./.code-search search query

# Per-token diagnostics
rs-search explain "tokenize identifier"

# MCP stdio server
rs-search serve
```

Positional queries also work without the `search` keyword (`rs-search foo bar`).

---

## MCP server

`rs-search serve` speaks JSON-RPC 2.0 over stdio. It exposes a `search` tool whose arguments can override project root, index dir, and discipline on a **per-call** basis — one server can index many roots.

- Tool input schema caps `query` length at 8192 chars.
- `tools/call` is wrapped in `std::panic::catch_unwind`; handler panics emit a JSON-RPC `-32603` error instead of killing the session.
- Results stream only **after** the scan + embed + sweep pass completes — no partial snapshots.

---

## Architecture

```
            ┌────────────────────────────┐
  query ──▶ │ tokenize (camel/kebab/snake)│──┐
            └────────────────────────────┘  │
                                             ▼
                                  ┌──────────────────┐
                                  │ BM25 (lexical)   │──┐
                                  └──────────────────┘  │
                                                        ▼
                                              ┌──────────────────┐
                                              │ RRF fusion (k=60)│──▶ results
                                              └──────────────────┘
                                                        ▲
                                  ┌──────────────────┐  │
                                  │ vector rerank    │──┘
                                  │ (nomic-embed)    │
                                  └──────────────────┘
```

### Scanner (`src/scanner.rs`)

Uses `ignore::WalkBuilder` (the ripgrep / fd crate). Honors `.gitignore`, `.git/info/exclude`, global gitignore, and `.codesearchignore` out of the box, with an extra `IGNORED_DIRS` allowlist for vendored caches. Files are split into ~40-line overlapping chunks. PDF files dispatch to `src/pdf.rs`.

### Embedder (`src/embed.rs`)

- Model file: `nomic-embed-text-v1.5.Q4_K_M.gguf` (split across 6 part files under `models/`)
- Loader: `gguf_file::Content::read` → dequantize → `VarBuilder::from_tensors` → `NomicBertModel`
- Mean-pool + L2-normalize on the way out
- Tokenizer: BERT WordPiece from a bundled `models/tokenizer.json` (`include_bytes!`)
- Lazy: initialized via `OnceLock<Result<Embedder, String>>` on first query
- SIMD cosine via `simsimd::SpatialSimilarity::cosine` with scalar fallback

### Fusion (`src/fusion.rs`)

Reciprocal Rank Fusion with `RRF_K = 60`. `looks_like_identifier(query)` (snake / kebab / dotted / camel without spaces) triggers a `1.5×` weight on the BM25 rank. Output normalized to `[0, 1]`.

### Embedding cache (`src/embed_cache.rs`)

Content-addressable, keyed by `BLAKE3(model_tag || dim || text)`. Two tiers:

- In-memory `Mutex<HashMap>` per process
- On-disk `f32` blobs under `<index>/emb-cache/<hex>.bin`

Dim is part of the key, so Matryoshka truncation (`RS_SEARCH_DIM=256`) gets its own cache lane. Orphan sweep runs at end of each full search.

### PDF ingest (`src/pdf.rs`)

- Crate: `pdf-extract = "0.9"` (pure Rust, no C deps)
- Splits on form-feed (`\x0c`) → one `Chunk` per page
- Each page chunk has `line_start = line_end = page_number`
- Cache at `<index>/pdf-cache/<hash>.json` keyed on `abs_path + mtime`
- Honors the 50 MB file cap; encrypted / scanned-only / malformed PDFs yield zero chunks silently (no OCR)

---

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `RS_SEARCH_DIM` | full | Matryoshka truncation — slice the vector to N dims and renormalize |
| `RS_SEARCH_QUERY_PREFIX` | `search_query: ` | Embedder prefix applied to query text |
| `RS_SEARCH_DOC_PREFIX` | `search_document: ` | Embedder prefix applied to chunk text |

The defaults match `nomic-embed-text-v1.5`. For CodeRankEmbed swap the query prefix to `Represent this query for searching relevant code: `.

---

## Feature gates

| Feature | Default | Pulls in |
|---|---|---|
| `vector` | yes | `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`, `libsql`, `tokio` |
| `perf` | yes | `mimalloc` as `#[global_allocator]` |
| `pdf` | yes | `pdf-extract` |
| `simd` | yes | `simsimd` SIMD cosine |

Disabling `vector` shrinks the binary to a pure-Rust BM25 + RRF + PDF scanner.

---

## Build constraints

### Nightly required

`candle-core` 0.10 uses `usize::is_multiple_of()` (rust-lang/rust#128101), unstable under `unsigned_is_multiple_of`. Stable rustc rejects this with `E0658` even on 1.94.1. The project pins nightly via `rust-toolchain.toml`; CI uses `dtolnay/rust-toolchain@nightly`.

### Windows: MSVC

`candle-core` → `tokenizers[onig]` → `oniguruma` (C). The C build runs through MSVC, not MinGW. Local Windows dev needs Visual Studio Build Tools with the C++ workload (`cl.exe` on `PATH`). `windows-latest` runners have it pre-installed.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for explicit `RUSTC` / `CC` / `INCLUDE` / `LIB` env setup.

---

## Eval harness

`src/eval.rs` exports `ndcg_at_k`, `mrr`, `recall_at_k`, `precision_at_k`, and an `EvalReport` aggregator — plug in BEIR / CoIR qrels to gate NDCG@10 regressions in CI.

---

## Releases

- [`release-plz.toml`](./release-plz.toml) — conventional-commit-driven changelog and version-bump PRs
- `[workspace.metadata.dist]` in `Cargo.toml` — cross-platform binaries via `cargo-dist`
- [`.github/workflows/`](./.github/workflows) — `build.yml`, `cascade.yml`, `gh-pages.yml`, `auto-declaudeify.yml`

---

## License

See repository.
