# rs-search

A small Rust crate compiled to a WebAssembly `cdylib`. It provides the search-fusion and text-analysis pieces consumed inside the plugkit WASM stack. It is not a standalone tool: there is no CLI, no MCP server, no embedding model, and no installable binary.

The crate ranks results by Reciprocal Rank Fusion over candidate lists supplied by the WASM host, plus supporting tokenization, enclosing-context detection, and offline evaluation metrics.

## Dependencies

`serde`, `serde_json`, `regex`. Nothing else.

## What it does

### Fusion (`src/fusion.rs`)

Reciprocal Rank Fusion with `RRF_K = 60`. The live fused path is `rrf_merge_n`, which merges any number of ranked id lists with equal weight and returns raw RRF scores.

A weighted variant (`fuse` / `rrf_merge`) that applies a `1.5x` BM25 boost for identifier-shaped queries (`looks_like_identifier`, gated by `IDENTIFIER_BOOST`) and a `normalize_scores` helper that scales scores to `[0, 1]` also exist in this module, but are NOT yet wired into the live fused path; wiring them requires a measured search-quality evaluation first.

### Tokenization (`src/tokenize.rs`)

`tokenize` splits text into lowercased tokens, with identifier-aware splitting: camelCase boundaries (`split_camel`) plus kebab/snake/dot separators. Tokens shorter than 2 chars are dropped; output is deduplicated and sorted.

### Enclosing context (`src/context.rs`)

`find_enclosing_context` scans upward from a target line to find the nearest enclosing function, class, struct, or `impl` name, used to label snippets. A regex matches `function`/`class`/`const|let|var = (`/`fn`/`struct`/`impl` declarations and skips language keywords. `get_file_total_lines` reads a file relative to a root and returns its line count.

### Evaluation metrics (`src/eval.rs`)

Offline ranking metrics against a qrels map: `ndcg_at_k`, `mrr`, `recall_at_k`, `precision_at_k`, plus `dcg`. `evaluate` aggregates NDCG@10, MRR, Recall@100, and P@10 into an `EvalReport`, and `format_report` renders it as text. Plug in qrels to gate ranking-quality regressions.

### Host boundary (`src/wasm_host.rs`)

Declares the host imports under `wasm_import_module = "env"`: `host_vec_search`, `host_bm25_search`, `host_git_search`, `host_log`, `host_now_ms`. Each returns a packed `(ptr, len)` `u64` that is unpacked and decoded as a JSON `Vec<HostHit>`.

`fusion_search(query, k, root)` is the live entry: it asks the host for vector, BM25, and git candidates (candidate count is `k * 5` floored at 50), collects their payloads, merges the three ranked lists with `rrf_merge_n`, takes the top `k`, and sorts by score then id.

### Public API (`src/lib.rs`)

`Searcher::new(root)` and `Searcher::search(query, k)` drive `fusion_search` and map the host hits into `SearchHit { id, score, snippet }`, pulling `snippet` out of each hit payload. `k == 0` returns empty.

## Build and ship

This is a WASM `cdylib` (`crate-type = ["cdylib"]`, `wasm` feature). It builds and ships through the WASM cascade: pushing to this repo triggers `.github/workflows/cascade.yml`, and `.github/workflows/wasm-check.yml` validates the WASM build. No local `cargo build`/`cargo install`, no published binary, no standalone install.

## License

See repository.
