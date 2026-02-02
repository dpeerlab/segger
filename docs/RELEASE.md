# Release Process

This checklist standardizes releases and keeps code, docs, and tags in sync.

## 1. Prepare the Release

- Ensure `main` is green and has the intended changes.
- Update `pyproject.toml` version.
- Update `CHANGELOG.md` with release notes and date.
- Review docs that mention the version (README, docs, slides if needed).

## 2. Run Tests

CPU-only (local or CI):

```bash
PYTHONPATH=src pytest tests/ -v -m "not gpu and not spatialdata and not sopa" \
  --ignore=tests/test_spatialdata_io.py
```

SpatialData (optional dependency):

```bash
PYTHONPATH=src pytest tests/test_spatialdata_io.py -v
```

GPU tests (when CUDA + RAPIDS are available):

```bash
PYTHONPATH=src pytest tests/test_prediction_graph.py -v
PYTHONPATH=src pytest tests/test_alignment_loss.py -v
PYTHONPATH=src pytest tests/test_alignment_loss_integration.py -v
```

## 3. Build and Verify

```bash
python -m build
python -m pip install dist/segger-*.whl
python -c "import segger; print('segger import ok')"
```

## 4. Tag and Publish

```bash
git tag -a vX.Y.Z -m "Segger vX.Y.Z"
git push origin vX.Y.Z
```

Then create a GitHub Release using the changelog entry.

## 5. Post-Release

- Bump to the next development version if desired (e.g., `0.2.1-dev`).
- Open a tracking issue for the next milestone.
