# Contributing to rs-search

## Local build (Windows)

rs-search depends on `onig_sys` and `candle-core` nightly features. The Chocolatey `cargo.exe` shim uses stable rustc even when `RUSTUP_TOOLCHAIN` is set — it does NOT work.

Use the rustup nightly binary directly:

```pwsh
$env:RUSTC = "$env:USERPROFILE\.rustup\toolchains\nightly-x86_64-pc-windows-msvc\bin\rustc.exe"
$env:PATH  = "$env:USERPROFILE\.rustup\toolchains\nightly-x86_64-pc-windows-msvc\bin;$env:PATH"
& "$env:USERPROFILE\.rustup\toolchains\nightly-x86_64-pc-windows-msvc\bin\cargo.exe" build --release
```

### MSVC env for onig_sys

Set `CC` to the MSVC `cl.exe` and extend `INCLUDE` / `LIB` to cover MSVC + Windows SDK headers, e.g.:

```pwsh
$env:CC      = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe"
$env:INCLUDE = "<MSVC>\include;<WinSDK>\Include\<ver>\ucrt;<WinSDK>\Include\<ver>\shared;<WinSDK>\Include\<ver>\um"
$env:LIB     = "<MSVC>\lib\x64;<WinSDK>\Lib\<ver>\ucrt\x64;<WinSDK>\Lib\<ver>\um\x64"
```

### Switching toolchains

Always `cargo clean` when switching between stable and nightly — artifact incompatibility produces `E0514: found crate compiled by an incompatible version of rustc`.

## Quick install

Prefer letting GitHub Actions build. To install from source without cloning:

```bash
cargo install --git https://github.com/AnEntrypoint/rs-search rs-search
```

## Build features

- `vector` (default) — candle-core + GGUF embedding loader
- `perf` (default) — mimalloc allocator
- `pdf` (default) — pdf-extract page-level indexing

Print what's enabled at runtime:

```
rs-search --features
```

## CI

Cross-platform builds happen via `.github/workflows/release.yml`. Local builds are optional; CI is authoritative.
