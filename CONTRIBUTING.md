# Contributing to rs-search

## Local build

This is a plain stable-Rust crate (`serde`, `serde_json`, `regex` only) compiled to a WASM `cdylib`. No nightly toolchain, no MSVC/oniguruma setup, no GGUF model download.

```bash
cargo check --target wasm32-wasip1 --no-default-features --features wasm --lib
cargo test --lib
```

`rustup target add wasm32-wasip1` first if the target isn't installed.

## Build features

- `wasm` — the `cdylib` wasm build feature.
- default features: none.

## Ship

There is no standalone binary and no `cargo install` path. Pushing to this repo triggers `.github/workflows/cascade.yml` and `.github/workflows/wasm-check.yml`; CI is authoritative and runs on `dtolnay/rust-toolchain@stable`.

## Testing

No synthetic test files or mocking frameworks. Verify via `cargo test --lib` (the in-module `#[cfg(test)]` blocks in `src/fusion.rs`) and the root `test.js` (structural/hygiene checks, mock-free, real file reads: `node test.js`).
