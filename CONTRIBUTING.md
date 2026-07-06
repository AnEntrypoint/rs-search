# Contributing to rs-search

## Local build

This is a plain stable-Rust, dependency-free `rlib` crate. No nightly toolchain, no wasm target, no feature flags, no MSVC/oniguruma setup, no GGUF model download.

```bash
cargo build --lib
cargo check --lib
```

## Ship

There is no standalone binary and no `cargo install` path. Pushing to this repo triggers `.github/workflows/cascade.yml`; CI is authoritative and runs on the stable toolchain.

## Testing

No synthetic test files, no `#[cfg(test)]` blocks, no mocking frameworks. Verify via the root `test.js` (structural/hygiene checks, mock-free, real file reads: `node test.js`) plus real compiled execution (`cargo build --lib`, or a throwaway `cargo run --example` witness deleted after use) for behavioral changes.
