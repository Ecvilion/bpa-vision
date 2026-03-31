"""Tests for calibration homography diagnostics."""

from __future__ import annotations

import pytest

from tools.calibration.homography import _compute_homography_impl, compute_homography


class TestComputeHomography:
    def test_returns_rich_diagnostics_for_exact_mapping(self):
        src = [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]
        dst = [
            [100.0, 50.0],
            [140.0, 50.0],
            [140.0, 90.0],
            [100.0, 90.0],
        ]

        result = _compute_homography_impl(src, dst)

        assert len(result.matrix) == 3
        assert len(result.reprojected_points) == 4
        assert len(result.inlier_mask) == 4
        assert result.inliers == 4
        assert all(result.inlier_mask)
        assert result.reprojection_error == pytest.approx(0.0, abs=1e-4)
        assert result.inlier_reprojection_error == pytest.approx(0.0, abs=1e-4)
        assert result.source_coverage == pytest.approx(1.0)
        assert result.destination_coverage == pytest.approx(1.0)

    def test_compute_wrapper_keeps_legacy_contract(self):
        src = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        dst = [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]

        matrix, error = compute_homography(src, dst)

        assert len(matrix) == 3
        assert error == pytest.approx(0.0, abs=1e-4)

    def test_all_points_method_uses_every_pair(self):
        src = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]]
        dst = [[100.0, 50.0], [140.0, 50.0], [140.0, 90.0], [100.0, 90.0], [120.0, 70.0]]

        result = _compute_homography_impl(src, dst, homography_method="all_points")

        assert result.estimator == "ALL_POINTS"
        assert result.inliers == 5
        assert result.inlier_mask == [True, True, True, True, True]

    def test_rejects_duplicate_source_points(self):
        src = [[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        dst = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

        with pytest.raises(ValueError, match="unique source points"):
            _compute_homography_impl(src, dst)

    def test_rejects_invalid_runtime_parameters(self):
        src = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        dst = [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]

        with pytest.raises(ValueError, match="ransac_reproj_threshold"):
            _compute_homography_impl(src, dst, ransac_reproj_threshold=0.0)

        with pytest.raises(ValueError, match="confidence"):
            _compute_homography_impl(src, dst, confidence=0.0)

        with pytest.raises(ValueError, match="max_iterations"):
            _compute_homography_impl(src, dst, max_iterations=0)

        with pytest.raises(ValueError, match="Unsupported homography_method"):
            _compute_homography_impl(src, dst, homography_method="bogus")
