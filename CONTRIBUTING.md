# Contributing to rs-search

## Local build

This is a plain stable-Rust, dependency-free `rlib` crate. No nightly toolchain, no wasm target, no feature flags, no MSVC/oniguruma setup, no GGUF model download.

```bash
cargo build --lib
cargo check --lib
```

## Ship

There is no standalone binary and no `cargo install` path. Pushing to this repo triggers `.github/workflows/cascade.yml`; CI is authoritative and runs on the stable toolchain.

## Verification

No test files of any kind, no `#[cfg(test)]` blocks, no mocking frameworks. Verify via real compiled execution only: `cargo build --lib`, `cargo check --lib`, or a throwaway `cargo run --example` witness deleted after use for behavioral changes. Manual troubleshooting and debugging against the real crate is the entire verification surface.
