"""Tests for platform-specific quality filtering.

This module tests the quality filtering functionality for different
spatial transcriptomics platforms (Xenium, CosMx, MERSCOPE).

All tests are CPU-only and don't require GPU or optional dependencies.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from segger.io.quality_filter import (
    XeniumQualityFilter,
    CosMxQualityFilter,
    MerscopeQualityFilter,
    SpatialDataQualityFilter,
    get_quality_filter,
    filter_transcripts,
)


class TestXeniumQualityFilter:
    """Tests for Xenium QV-based quality filtering."""

    def test_filter_by_qv_threshold(self, transcripts_with_qv_range: pl.DataFrame):
        """Test QV threshold filtering."""
        qf = XeniumQualityFilter()
        df = transcripts_with_qv_range.lazy()

        # Filter with default threshold (20)
        filtered = qf.filter(df, min_threshold=20.0, feature_column="feature_name").collect()

        # All remaining transcripts should have QV >= 20
        assert filtered["qv"].min() >= 20.0
        # Original has 5 transcripts with QV < 20 (5, 10, 15, 18, 19)
        assert len(filtered) == len(transcripts_with_qv_range) - 5

    def test_filter_strict_threshold(self, transcripts_with_qv_range: pl.DataFrame):
        """Test stricter QV threshold (30)."""
        qf = XeniumQualityFilter()
        df = transcripts_with_qv_range.lazy()

        filtered = qf.filter(df, min_threshold=30.0, feature_column="feature_name").collect()

        assert filtered["qv"].min() >= 30.0
        # Count transcripts with QV >= 30 in original
        expected = len(transcripts_with_qv_range.filter(pl.col("qv") >= 30))
        assert len(filtered) == expected

    def test_filter_no_threshold(self, transcripts_with_qv_range: pl.DataFrame):
        """Test with no QV threshold (only control probe filtering)."""
        qf = XeniumQualityFilter()
        df = transcripts_with_qv_range.lazy()

        # No QV filtering when threshold is None or 0
        filtered = qf.filter(df, min_threshold=None, feature_column="feature_name").collect()
        assert len(filtered) == len(transcripts_with_qv_range)

        filtered = qf.filter(df, min_threshold=0, feature_column="feature_name").collect()
        assert len(filtered) == len(transcripts_with_qv_range)

    def test_filter_control_probes(self, transcripts_with_control_probes: pl.DataFrame):
        """Test control probe removal for Xenium."""
        qf = XeniumQualityFilter()
        df = transcripts_with_control_probes.lazy()

        # Filter without QV threshold to test probe removal only
        filtered = qf.filter(df, min_threshold=None, feature_column="feature_name").collect()

        # Control probes should be removed
        remaining_genes = filtered["feature_name"].to_list()
        assert "NegControlProbe_0001" not in remaining_genes
        assert "NegControlProbe_0002" not in remaining_genes
        assert "antisense_Gene1" not in remaining_genes
        assert "BLANK_0001" not in remaining_genes
        assert "BLANK_0002" not in remaining_genes

        # Real genes should remain
        assert "Gene1" in remaining_genes
        assert "Gene2" in remaining_genes

    def test_quality_column_property(self):
        """Test quality_column property returns correct value."""
        qf = XeniumQualityFilter()
        assert qf.quality_column == "qv"

    def test_platform_name_property(self):
        """Test platform_name property."""
        qf = XeniumQualityFilter()
        assert qf.platform_name == "Xenium"


class TestCosMxQualityFilter:
    """Tests for CosMx control probe filtering."""

    def test_filter_control_probes(self, transcripts_with_control_probes: pl.DataFrame):
        """Test CosMx control probe removal."""
        qf = CosMxQualityFilter()
        df = transcripts_with_control_probes.lazy()

        filtered = qf.filter(df, min_threshold=None, feature_column="feature_name").collect()

        remaining_genes = filtered["feature_name"].to_list()

        # CosMx patterns should be removed
        assert "Negative_Control_1" not in remaining_genes
        assert "SystemControl_0001" not in remaining_genes
        assert "NegPrb_001" not in remaining_genes

        # Real genes should remain
        assert "Gene1" in remaining_genes
        assert "Gene4" in remaining_genes

    def test_qv_threshold_warning(self, transcripts_with_qv_range: pl.DataFrame):
        """Test that providing QV threshold emits a warning."""
        qf = CosMxQualityFilter()
        df = transcripts_with_qv_range.lazy()

        with pytest.warns(UserWarning, match="does not have per-transcript quality scores"):
            qf.filter(df, min_threshold=20.0, feature_column="feature_name").collect()

    def test_no_quality_column(self):
        """Test that quality_column returns None."""
        qf = CosMxQualityFilter()
        assert qf.quality_column is None

    def test_platform_name_property(self):
        """Test platform_name property."""
        qf = CosMxQualityFilter()
        assert qf.platform_name == "CosMx"


class TestMerscopeQualityFilter:
    """Tests for MERSCOPE blank barcode filtering."""

    def test_filter_blank_barcodes(self, transcripts_with_control_probes: pl.DataFrame):
        """Test MERSCOPE blank barcode removal."""
        qf = MerscopeQualityFilter()
        df = transcripts_with_control_probes.lazy()

        filtered = qf.filter(df, min_threshold=None, feature_column="feature_name").collect()

        remaining_genes = filtered["feature_name"].to_list()

        # BLANK patterns should be removed
        assert "BLANK_0001" not in remaining_genes
        assert "BLANK_0002" not in remaining_genes

        # Real genes should remain
        assert "Gene1" in remaining_genes
        # Note: Other control probes are NOT removed by MERSCOPE filter
        # (they're Xenium/CosMx specific)

    def test_qv_threshold_warning(self, transcripts_with_qv_range: pl.DataFrame):
        """Test that providing QV threshold emits a warning."""
        qf = MerscopeQualityFilter()
        df = transcripts_with_qv_range.lazy()

        with pytest.warns(UserWarning, match="does not have per-transcript quality scores"):
            qf.filter(df, min_threshold=20.0, feature_column="feature_name").collect()

    def test_no_quality_column(self):
        """Test that quality_column returns None."""
        qf = MerscopeQualityFilter()
        assert qf.quality_column is None

    def test_calculate_fdr(self):
        """Test FDR calculation from blank barcodes."""
        qf = MerscopeQualityFilter()

        # Create data with known blank ratio
        df = pl.DataFrame({
            "gene": ["Gene1", "Gene2", "BLANK_001", "Gene3", "BLANK_002"],
            "x": [0, 1, 2, 3, 4],
            "y": [0, 1, 2, 3, 4],
        }).lazy()

        fdr = qf.calculate_fdr(df, feature_column="gene")

        # 2 blanks out of 5 total = 0.4 FDR
        assert abs(fdr - 0.4) < 0.01

    def test_calculate_fdr_no_blanks(self):
        """Test FDR calculation with no blank barcodes."""
        qf = MerscopeQualityFilter()

        df = pl.DataFrame({
            "gene": ["Gene1", "Gene2", "Gene3"],
            "x": [0, 1, 2],
            "y": [0, 1, 2],
        }).lazy()

        fdr = qf.calculate_fdr(df, feature_column="gene")
        assert fdr == 0.0

    def test_platform_name_property(self):
        """Test platform_name property."""
        qf = MerscopeQualityFilter()
        assert qf.platform_name == "MERSCOPE"


class TestQualityFilterHelpers:
    """Tests for helper utilities and dispatch logic."""

    def test_get_quality_filter_aliases(self):
        """Test that platform aliases resolve to correct filter classes."""
        assert isinstance(get_quality_filter("xenium"), XeniumQualityFilter)
        assert isinstance(get_quality_filter("10x_xenium"), XeniumQualityFilter)
        assert isinstance(get_quality_filter("cosmx"), CosMxQualityFilter)
        assert isinstance(get_quality_filter("nanostring_cosmx"), CosMxQualityFilter)
        assert isinstance(get_quality_filter("merscope"), MerscopeQualityFilter)
        assert isinstance(get_quality_filter("vizgen_merscope"), MerscopeQualityFilter)
        assert isinstance(get_quality_filter("spatialdata"), SpatialDataQualityFilter)

    def test_get_quality_filter_invalid_platform(self):
        """Test that invalid platform raises a helpful error."""
        with pytest.raises(ValueError, match="Unknown platform"):
            get_quality_filter("not-a-platform")

    def test_filter_transcripts_dispatch_xenium(self, transcripts_with_qv_range: pl.DataFrame):
        """Test convenience filter_transcripts uses Xenium QV thresholding."""
        df = transcripts_with_qv_range.lazy()
        filtered = filter_transcripts(df, platform="xenium", min_qv=25.0).collect()
        assert filtered["qv"].min() >= 25.0

    def test_filter_transcripts_dispatch_cosmx(self, transcripts_with_control_probes: pl.DataFrame):
        """Test convenience filter_transcripts removes CosMx control probes."""
        df = transcripts_with_control_probes.lazy()
        filtered = filter_transcripts(df, platform="cosmx").collect()
        remaining_genes = filtered["feature_name"].to_list()
        assert "Negative_Control_1" not in remaining_genes
        assert "SystemControl_0001" not in remaining_genes
        assert "NegPrb_001" not in remaining_genes


class TestSpatialDataQualityFilter:
    """Tests for SpatialData platform auto-detection."""

    def test_auto_detect_xenium(self):
        """Test auto-detection of Xenium from QV column."""
        qf = SpatialDataQualityFilter()

        df = pl.DataFrame({
            "feature_name": ["Gene1", "Gene2"],
            "qv": [30.0, 25.0],
            "x": [0, 1],
            "y": [0, 1],
        }).lazy()

        # Should detect Xenium from qv column
        filtered = qf.filter(df, min_threshold=25.0, feature_column="feature_name").collect()
        assert len(filtered) == 2  # Both Gene1 (30) and Gene2 (25) have QV >= 25

    def test_auto_detect_cosmx(self):
        """Test auto-detection of CosMx from CellComp column."""
        qf = SpatialDataQualityFilter()

        df = pl.DataFrame({
            "feature_name": ["Gene1", "Negative_Ctrl"],
            "CellComp": ["Nuclear", "Nuclear"],
            "x": [0, 1],
            "y": [0, 1],
        }).lazy()

        # Should detect CosMx from CellComp column
        filtered = qf.filter(df, min_threshold=None, feature_column="feature_name").collect()
        assert "Negative_Ctrl" not in filtered["feature_name"].to_list()

    def test_explicit_platform(self):
        """Test explicit platform specification."""
        qf = SpatialDataQualityFilter(platform="xenium")

        df = pl.DataFrame({
            "feature_name": ["Gene1", "BLANK_001"],
            "qv": [30.0, 25.0],
            "x": [0, 1],
            "y": [0, 1],
        }).lazy()

        filtered = qf.filter(df, min_threshold=None, feature_column="feature_name").collect()
        # BLANK should be filtered by Xenium filter
        assert "BLANK_001" not in filtered["feature_name"].to_list()

    def test_unknown_platform_warning(self):
        """Test warning when platform cannot be detected."""
        qf = SpatialDataQualityFilter()

        df = pl.DataFrame({
            "feature_name": ["Gene1", "Gene2"],
            "x": [0, 1],
            "y": [0, 1],
        }).lazy()

        with pytest.warns(UserWarning, match="Could not detect platform"):
            filtered = qf.filter(df, min_threshold=None, feature_column="feature_name").collect()

        # Should return unfiltered data
        assert len(filtered) == 2


class TestGetQualityFilter:
    """Tests for the get_quality_filter factory function."""

    def test_get_xenium_filter(self):
        """Test getting Xenium filter by name."""
        qf = get_quality_filter("xenium")
        assert isinstance(qf, XeniumQualityFilter)

        qf = get_quality_filter("10x_xenium")
        assert isinstance(qf, XeniumQualityFilter)

    def test_get_cosmx_filter(self):
        """Test getting CosMx filter by name."""
        qf = get_quality_filter("cosmx")
        assert isinstance(qf, CosMxQualityFilter)

        qf = get_quality_filter("nanostring_cosmx")
        assert isinstance(qf, CosMxQualityFilter)

    def test_get_merscope_filter(self):
        """Test getting MERSCOPE filter by name."""
        qf = get_quality_filter("merscope")
        assert isinstance(qf, MerscopeQualityFilter)

        qf = get_quality_filter("vizgen_merscope")
        assert isinstance(qf, MerscopeQualityFilter)

    def test_get_spatialdata_filter(self):
        """Test getting SpatialData filter by name."""
        qf = get_quality_filter("spatialdata")
        assert isinstance(qf, SpatialDataQualityFilter)

    def test_unknown_platform_error(self):
        """Test error for unknown platform."""
        with pytest.raises(ValueError, match="Unknown platform"):
            get_quality_filter("unknown_platform")

    def test_case_insensitive(self):
        """Test that platform names are case-insensitive."""
        qf1 = get_quality_filter("Xenium")
        qf2 = get_quality_filter("XENIUM")
        qf3 = get_quality_filter("xenium")

        assert type(qf1) == type(qf2) == type(qf3)


class TestFilterTranscripts:
    """Tests for the filter_transcripts convenience function."""

    def test_filter_transcripts_xenium(self, transcripts_with_qv_range: pl.DataFrame):
        """Test convenience function with Xenium."""
        df = transcripts_with_qv_range.lazy()

        filtered = filter_transcripts(
            df, platform="xenium", min_qv=20.0, feature_column="feature_name"
        ).collect()

        assert filtered["qv"].min() >= 20.0

    def test_filter_transcripts_cosmx(self, transcripts_with_control_probes: pl.DataFrame):
        """Test convenience function with CosMx."""
        df = transcripts_with_control_probes.lazy()

        # Should warn about min_qv being ignored
        with pytest.warns(UserWarning):
            filtered = filter_transcripts(
                df, platform="cosmx", min_qv=20.0, feature_column="feature_name"
            ).collect()

        assert "Negative_Control_1" not in filtered["feature_name"].to_list()


class TestIntegrationWithToyData:
    """Integration tests using the toy Xenium dataset."""

    def test_xenium_filter_on_toy_data(self, toy_transcripts: pl.DataFrame):
        """Test Xenium filter on realistic toy data."""
        qf = XeniumQualityFilter()
        df = toy_transcripts.lazy()

        # Filter with threshold
        filtered = qf.filter(
            df, min_threshold=20.0, feature_column="feature_name", quality_column="qv"
        ).collect()

        # Should have fewer transcripts
        assert len(filtered) < len(toy_transcripts)

        # Control probes should be removed
        remaining_genes = set(filtered["feature_name"].to_list())
        assert not any(g.startswith("NegControl") for g in remaining_genes)
        assert not any(g.startswith("BLANK_") for g in remaining_genes)

    def test_no_filter_preserves_all(self, toy_transcripts: pl.DataFrame):
        """Test that no filtering preserves all non-control transcripts."""
        qf = XeniumQualityFilter()
        df = toy_transcripts.lazy()

        # Get count without control probes
        control_patterns = ["NegControl", "antisense_", "BLANK_"]
        non_control = toy_transcripts.filter(
            ~pl.col("feature_name").str.contains("|".join(control_patterns))
        )

        # Filter without QV threshold
        filtered = qf.filter(
            df, min_threshold=None, feature_column="feature_name"
        ).collect()

        assert len(filtered) == len(non_control)
