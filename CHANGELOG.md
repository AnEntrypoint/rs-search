# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0](https://github.com/AnEntrypoint/rs-search/releases/tag/v0.1.0) - 2026-04-21

### Bug Fixes

- *(embed)* remap llama.cpp GGUF tensor names to candle NomicBert HF keys
- *(embed)* borrow PathBuf when constructing EmbedCache
- prune ignored dirs early via WalkBuilder.filter_entry + suffix/prefix wildcard tables for browser profiles
- raise file size limit from 5MB to 50MB to index large data files
- add .json to CODE_EXTENSIONS so JSON files are indexed by search
- add .plugkit-browser-profile to ignored dirs, add .prd to code extensions
- replace lookahead regex with stdlib camelCase split, OnceLock symbol regexes
- resolve borrow error in scanner chunking

### CI

- update auto-declaudeify workflow
- add auto-declaudeify workflow

### Documentation

- reflect RRF fusion, feature gates, MRL, subcommands, release tooling

### Features

- gate pdf-extract behind 'pdf' feature (default on)
- *(embed)* bundle GGUF model via include_bytes! — vector search always works
- architectural rework for perf, quality, and release hygiene
- index PDFs page-by-page for search
- 4-section search output with git commit indexing
- add vector embedding search with nomic-embed-text-v1.5 GGUF
- add nomic-embed-text-v1.5.Q4_K_M split into 6 parts for vector search
- cascade to rs-plugkit on push to main
- add lib.rs public API for rs-plugkit integration
- initial Rust port of codebasesearch with BM25 and MCP support

### Refactor

- drop walkdir/anyhow/serde, use stdlib read_dir for pruning
