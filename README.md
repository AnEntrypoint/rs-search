# rs-search

A small, dependency-free Rust `rlib` (`rs_search`). It provides the search-fusion and tokenization pieces consumed by rs-plugkit's WASM host. It is not a standalone tool: there is no CLI, no MCP server, no embedding model, no installable binary, no cdylib target.

The crate has exactly three source files: `src/lib.rs`, `src/fusion.rs`, `src/tokenize.rs`. It ranks candidate id lists by Reciprocal Rank Fusion and tokenizes text for indexing/search.

## Dependencies

None. `Cargo.toml` has no `[dependencies]` section.

## What it does

### Fusion (`src/fusion.rs`)

Reciprocal Rank Fusion with `RRF_K = 60`. The entry point is `fuse_n(ranked_lists, weights, query)`: when `looks_like_identifier(query)` is true (query has a separator -- `_`/`-`/`.` -- or a mid-word case transition, e.g. `myVariable`/`HTTPServer`; a single Title-Case word like `Hello` does not qualify), it calls `rrf_merge_n_weighted(ranked_lists, weights)`, applying the caller's per-list weights (rs-plugkit passes `[1.0, IDENTIFIER_BOOST]` to boost the BM25 list for identifier-shaped queries). Otherwise it calls `rrf_merge_n(ranked_lists)`, which merges any number of ranked id lists with equal weight -- the `weights` argument is intentionally not applied on this branch, since the boost is specifically an identifier-search heuristic and has no justified rationale for natural-language queries. Within a single ranked list, a duplicate id is deduplicated to its first (best) rank -- a backend that emits the same id twice in one list contributes only one RRF term for it. Both merge functions return raw (non-normalized) RRF scores.

### Tokenization (`src/tokenize.rs`)

`tokenize` splits text into lowercased tokens. `add_word_tokens` per word: if the word contains an uppercase letter, `split_camel` breaks it at camelCase boundaries, and each resulting piece is further split on every non-alphanumeric character. Every word (camelCase or not) is also split on every non-alphanumeric character directly (not just `-`/`_`/`.`), so `foo::bar` yields both `foo` and `bar`. If the whole word is alphanumeric-or-underscore only (no other punctuation), the whole lowercased word is kept as an additional token, so `my_variable_name` is searchable both as its parts (`my`, `variable`, `name`) and as the literal identifier. Output is deduplicated and sorted.

### Public API (`src/lib.rs`)

`pub mod fusion; pub mod tokenize;` -- the crate exposes exactly these two modules and nothing else.

## Build and ship

This is a plain `rlib`, `crate-type = ["rlib"]`, consumed as a normal Cargo path/git dependency by rs-plugkit. `cargo build`/`cargo check` work locally with the standard stable toolchain, no target/feature flags. Pushing to this repo triggers `.github/workflows/cascade.yml` as part of the wider cascade pipeline; CI is authoritative.

## License

See repository.
