## 2026-05-12 ci: cascade restricted to push-to-main (PR merge only)

- `.github/workflows/cascade.yml`: removed `pull_request: [main]` trigger. Cascade must only fire on PR merge (i.e., push to main), not on PR open — open PRs were dispatching downstream `rs-plugkit/build.yml` runs against unmerged commits. Push-to-main remains the sole dispatcher of `rs-plugkit/release.yml`.
