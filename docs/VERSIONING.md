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

## Release Note Structure

Each shipped version should include a release note file under `docs/releases/`
named `vX.Y.Z.md` and use this structure:

1. **High-level**
   - Short user-facing summary of the release.
   - Focus on major features, behavior changes, and migration impact.
2. **Low-level**
   - Concise technical details grouped by subsystem (CLI, data/model, export, tests/docs).
   - Include defaults/flags/API details only when they affect behavior or compatibility.

This keeps release communication readable for both users and maintainers.
