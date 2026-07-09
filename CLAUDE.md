@AGENTS.md

# rs-search

A small, dependency-free Rust `rlib` (`rs_search`) consumed as a normal Cargo git dependency by rs-plugkit. It provides search-fusion and tokenization primitives only -- there is no CLI, no MCP server, no embedding model, no installable binary, no `cdylib` target.

## Dependencies

None. `Cargo.toml` has no `[dependencies]` section.

## Toolchain

Builds on stable Rust. `rust-toolchain.toml` pins `channel = "stable"`. CI (`dtolnay/rust-toolchain@stable` in `.github/workflows/wasm-check.yml`) validates both the `wasm32-wasip1` target build and the native `cargo test --lib` suite.

## Architecture

The crate has exactly three source files: `src/lib.rs`, `src/fusion.rs`, `src/tokenize.rs`.

### Fusion (`src/fusion.rs`)

Reciprocal Rank Fusion with `RRF_K = 60`. The entry point is `fuse_n(ranked_lists, weights, query)`: when `looks_like_identifier(query)` is true (query has a separator -- `_`/`-`/`.` -- or a mid-word case transition, e.g. `myVariable`/`HTTPServer`; a single Title-Case word like `Hello` does not qualify), it calls `rrf_merge_n_weighted(ranked_lists, weights)`, applying the caller's per-list weights (rs-plugkit passes `[1.0, IDENTIFIER_BOOST]` to boost the BM25 list for identifier-shaped queries). Otherwise it calls `rrf_merge_n(ranked_lists)`, which merges any number of ranked id lists with equal weight -- the `weights` argument is intentionally not applied on this branch, since the boost is specifically an identifier-search heuristic and has no justified rationale for natural-language queries. Within a single ranked list, a duplicate id is deduplicated to its first (best) rank -- a backend that emits the same id twice in one list contributes only one RRF term for it. Both merge functions return raw (non-normalized) RRF scores.

### Tokenization (`src/tokenize.rs`)

`tokenize` splits text into lowercased tokens. `add_word_tokens` per word: if the word contains an uppercase letter, `split_camel` breaks it at camelCase boundaries, and each resulting piece is further split on every non-alphanumeric character. Every word (camelCase or not) is also split on every non-alphanumeric character directly (not just `-`/`_`/`.`), so `foo::bar` yields both `foo` and `bar`. If the whole word is alphanumeric-or-underscore only (no other punctuation), the whole lowercased word is kept as an additional token, so `my_variable_name` is searchable both as its parts (`my`, `variable`, `name`) and as the literal identifier. Output is deduplicated and sorted.

### Public API (`src/lib.rs`)

`pub mod fusion; pub mod tokenize;` -- the crate exposes exactly these two modules and nothing else.

## Build and ship

`crate-type = ["rlib"]`, consumed as a normal Cargo git dependency (see rs-plugkit's `Cargo.toml`, `rs-search` entry pointing at this repo). `cargo build`/`cargo check` work locally with the standard stable toolchain, no target/feature flags. Pushing to this repo triggers `.github/workflows/cascade.yml` as part of the wider cascade pipeline; CI is authoritative.

## Live wiring

`rs-plugkit` DOES depend on this crate (`Cargo.toml`: `rs-search = { git = "...", package = "rs-search" }`) and actively calls both live modules: `rs_search::tokenize::tokenize`/`add_word_tokens` (code_index.rs's BM25 ranking + git-commit-rank fallback tokenization) and `rs_search::fusion::fuse_n`/`IDENTIFIER_BOOST` (wasm_dispatch.rs's codesearch fusion of vector+BM25 result lists). Both modules are live, wired, load-bearing dependencies of rs-plugkit's real codesearch verb -- not orphaned.

## Verification

No test files of any kind, no `#[cfg(test)]` blocks, no test framework. Verification is real compiled execution only: `cargo build --lib`, `cargo check --lib`, or a throwaway `cargo run --example` witness deleted after use. Manual troubleshooting and debugging against the real crate is the entire verification surface.
