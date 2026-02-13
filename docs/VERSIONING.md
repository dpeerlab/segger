# Versioning

Segger follows Semantic Versioning (SemVer 2.0.0):

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: Backward-incompatible changes to the public API, CLI, or file formats.
- **MINOR**: Backward-compatible features and improvements.
- **PATCH**: Backward-compatible bug fixes and small maintenance changes.

## Pre-1.0 Policy

Until `1.0.0`, Segger uses a stricter interpretation:

- **MINOR** may include breaking changes.
- **PATCH** remains backward-compatible for the supported surface area.

## Source of Truth

The package version is defined in:

- `pyproject.toml` (`[project].version`)

Any release must update this value, the changelog, and the release notes.

## Release Comparison Baseline (Required)

For release branches, changelog and release-note scope must be computed against the previous released baseline (for `v0.2.0`, this is `v0.1.0`), not just against the most recent commit batch.

Use:

```bash
git rev-parse <previous-release-ref>
git log --oneline <previous-release-ref>..<release-branch>
git diff --stat <previous-release-ref>...<release-branch>
```

Required metadata in release docs:
- Previous-release baseline hash/date.
- Release-branch snapshot hash/date used for summary.
- Commit count and file/line delta from the previous release baseline.
- Major vs minor classification by subsystem.

## Pre-releases

Use pre-release suffixes for release candidates or previews:

```
0.3.0-rc.1
0.3.0-beta.2
```

## Versioning Scope

When deciding version bumps, consider:

- Python API: public functions, classes, and module behaviors
- CLI: flags, defaults, config file formats
- Output schemas: parquet or Zarr output formats
- Model checkpoints and training configs

If a change requires user code or data migration, treat it as breaking.

## Major vs Minor Classification

When writing release notes against the previous release baseline, classify each subsystem explicitly:

- **Major changes**:
  - New commands or workflows.
  - New output formats or schema families.
  - New model/loss behavior that materially changes training or inference semantics.
  - New dependency families that affect install/runtime behavior.
  - New I/O pathways (for example, new platform/store support).
- **Minor changes**:
  - Bug fixes that keep the same user contract.
  - Performance/stability improvements without new conceptual workflows.
  - Default tuning, logging/diagnostics, docs, and test hardening.
  - Internal refactors that do not materially change user-facing behavior.

## Release Note Structure (Required)

Each shipped version should include a release note file under `docs/releases/`
named `vX.Y.Z.md` and use this structure:

1. **High-level**
   - Must be derived from `<previous-release-ref>...<release-branch>` comparison.
   - Focus on major features, behavior changes, and migration impact.
2. **Low-level**
   - Technical details grouped by subsystem (CLI, data/model, export, I/O, tests/docs).
   - Include concrete option names/defaults/API notes when they affect behavior or compatibility.

## Worked Example: v0.2.0 (`v0.1.0` baseline -> `release/v2-stable`)

Comparison snapshot used:
- `v0.1.0` baseline reference: `dd681a8` (`2025-12-17`, `pyproject.toml` version `0.1.0`)
- Release snapshot: `2c92b43` (`2026-02-13`)
- Delta: `33` commits, `76` files, `18,232` insertions, `321` deletions.

Major classifications for `v0.2.0`:
- CLI lifecycle expansion (`segment` early stopping/best-ckpt prediction, new `predict`, new `plot`, richer `export`).
- New export architecture and formats (`merged`, `spatialdata`, `anndata`, Xenium improvements, boundary strategy controls).
- New SpatialData input/output pathways and lightweight `.zarr` interoperability utilities.
- New alignment-loss training path and checkpoint metadata contract (`segger_vocab`, `segger_me_gene_pairs`).
- New quality-filter layer, 3D-aware graph construction controls, and fragment-mode post-processing.
- New optional dependency model, dataset helpers, CI pipeline, and comprehensive test suite.

Minor classifications for `v0.2.0`:
- Prediction graph scale-factor alignment fix and boundary robustness improvements.
- ME-gene caching, progress messaging, and sampling/default tuning.
- Parallel export fallback hardening and import/lazy-loading cleanup.
- CLI help polishing and repository housekeeping changes.

This policy keeps release communication complete, comparable across branches, and easier for users to trust.
