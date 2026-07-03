@AGENTS.md

# rs-search

A Rust crate compiled to a WebAssembly `cdylib`. It provides search-fusion and text-analysis primitives consumed inside the plugkit WASM stack. There is no CLI, no MCP server, no embedding model, and no installable binary in this repo.

## Dependencies

`serde`, `serde_json`, `regex`. Nothing else — no `candle-core`, no GGUF loading, no oniguruma/MSVC requirement.

## Toolchain

Builds on stable Rust. `rust-toolchain.toml` pins `channel = "stable"`. CI (`dtolnay/rust-toolchain@stable` in `.github/workflows/wasm-check.yml`) validates both the `wasm32-wasip1` target build and the native `cargo test --lib` suite.

## Architecture

### Host boundary (`src/wasm_host.rs`)

Declares host imports under `#[link(wasm_import_module = "env")]`: `host_vec_search`, `host_bm25_search`, `host_git_search`, `host_log`, `host_now_ms`. Each search import returns a packed `(ptr, len)` `u64`; `take_bytes` unpacks and copies the bytes (bounded, error-returning — a malformed host response surfaces as a `Result::Err`, never a raw slice trap), and the caller decodes it as a JSON `Vec<HostHit>`.

`fusion_search(query, k, root)` is the live entry point: it requests vector, BM25, and git candidate lists from the host (candidate count `k * 5` floored at 50), collects payloads, and merges the three ranked lists via `crate::fusion::fuse_n` with per-source weights (`[1.0, IDENTIFIER_BOOST, 1.0]`), taking the top `k`.

### Fusion (`src/fusion.rs`)

Reciprocal Rank Fusion with `RRF_K = 60`. `fuse_n` is the live fused path: it calls `rrf_merge_n_weighted` (applies the `1.5x` `IDENTIFIER_BOOST` to the BM25 list) when `looks_like_identifier(query)` is true, else `rrf_merge_n` (equal weight across all lists). `looks_like_identifier` treats snake_case/kebab-case/dotted/camelCase single-token queries as identifier-shaped, while excluding bare decimal-number tokens (e.g. `3.14`) from the dot-separator check.

### Tokenization (`src/tokenize.rs`)

`tokenize` splits text into lowercased tokens, with identifier-aware splitting: camelCase boundaries (`split_camel`) plus kebab/snake/dot separators. Tokens shorter than 2 chars are dropped; output is deduplicated and sorted.

### Enclosing context (`src/context.rs`)

`find_enclosing_context` scans upward from a target line to find the nearest enclosing function, class, struct, or `impl` name, used to label snippets.

### Evaluation metrics (`src/eval.rs`)

Offline ranking metrics against a qrels map: `ndcg_at_k`, `mrr`, `recall_at_k`, `precision_at_k`, plus `dcg`. `evaluate` aggregates NDCG@10, MRR, Recall@100, and P@10 into an `EvalReport`.

### Public API (`src/lib.rs`)

`Searcher::new(root)` and `Searcher::search(query, k)` drive `fusion_search` and map host hits into `SearchHit { id, score, snippet }`. `k == 0` returns empty.

## Build and ship

`crate-type = ["cdylib"]`, `wasm` feature. Ships through the WASM cascade: pushing to this repo triggers `.github/workflows/cascade.yml`; `.github/workflows/wasm-check.yml` validates both the wasm target build and the native test suite. No local `cargo build`/`cargo install`, no published binary, no standalone install.

## Cascade wiring note

As of this writing, `rs-plugkit` does not depend on this crate — it implements its own host-vec-search-based codesearch directly. This crate's fusion/tokenize/context/eval logic is not currently consumed by the live cascade; fix bugs and keep docs accurate here regardless, since orphaned wiring status does not make the crate's own correctness or documentation less real.

## Testing

No synthetic test files, no test framework. Verification is `cargo test --lib` against the in-module `#[cfg(test)]` blocks (real code, no mocks) plus the single root `test.js` (structural/hygiene checks, mock-free, real file reads).
