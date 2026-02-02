# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Alignment loss with ME gene constraints from scRNA references.
- Fragment mode for grouping unassigned transcripts.
- SpatialData Zarr I/O and export utilities.
- Synthetic dataset generators, optional-deps helpers, and expanded tests.

### Changed
- Prediction graph uses polygon `prediction_scale_factor` (expands/shrinks boundaries).
- Dependency guidance and install notes for reproducible GPU setups.

### Fixed
- Edge permutation now skips negative partition labels to avoid bincount crashes.

## [0.2.0] - TBD

### Added
- Multi-task loss with triplet, metric, and alignment components.
- Fragment mode for unassigned transcripts with CPU/GPU backends.
- SpatialData Zarr I/O for scverse/SOPA interoperability.
- Xenium Explorer export utilities.
- Synthetic Xenium dataset generation for tests.

### Changed
- Polars-based data processing for improved performance.
- PCA-based gene and cell embeddings.

### Fixed
- TBD
