# Offline compiler publication verification

Canonical branch: five-ideas-development. Baseline: b969ccd4990e9e2780d486e009b0325d4aa6309d.
Local mirror baseline: a3880d97edd788c18b6337c6eff7429a800fdea7.

Baseline: existing1600 review suite24 and production core hashes preserved.
Modified: target/CLI suite40, real32 source compile9 cases28 proposals0 qualified;
JSON round-trip and exact cross-environment artifact bytes checked. No fitting.
Canonical publisher records literal commands, stdout, stderr and exit statuses in
verification.json. No official or independent semantic qualification claimed.

Four verified roles at `/tmp/qore-qualified-boundary-publication-20261009/`:
- Modified artifact: `modified.tar.gz` (owned code, targets, contract and progress).
- Diff: `patch.diff` (reopened and applied to an isolated restored baseline).
- Verification: `verification.json` (source/core/review hashes and command results).
- Runnable rollback: `rollback.py` plus `operation.json`/`original/` (hash guarded;
  default check-only, `--root <scratch> --apply` tested; live tree not rolled back).

Local role mirror: `C:/Users/14630/AppData/Local/Temp/qore-qualified-target-verification-20261009/`.
The publisher stops before commit if any mandatory command or integrity check fails.
GitHub commit/publication revision is captured in publication.json, outside source.
