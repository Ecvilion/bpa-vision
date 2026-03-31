"""Tests for experimental line-based distortion fitting."""

from __future__ import annotations

import pytest

from tools.calibration.line_based import (
    _line_rms_error,
    compute_line_based_distortion,
    distort_points_line_based,
    undistort_points_line_based,
)


def test_line_based_fit_recovers_synthetic_k1():
    image_width = 1280
    image_height = 720
    reference_model = {
        "model": "line_based_radial_v1",
        "k1": 0.72,
        "center_norm": [0.5, 0.5],
        "source_image_width": image_width,
        "source_image_height": image_height,
    }
    straight_lines = [
        [[180.0, 140.0], [360.0, 145.0], [560.0, 150.0], [760.0, 158.0], [980.0, 165.0]],
        [[260.0, 520.0], [400.0, 430.0], [540.0, 340.0], [680.0, 250.0], [820.0, 165.0]],
        [[1020.0, 610.0], [960.0, 500.0], [900.0, 390.0], [840.0, 280.0], [780.0, 170.0]],
    ]
    distorted_lines = [
        distort_points_line_based(
            line,
            model=reference_model,
            image_width=image_width,
            image_height=image_height,
        )
        for line in straight_lines
    ]

    result = compute_line_based_distortion(
        distorted_lines,
        image_width=image_width,
        image_height=image_height,
    )

    assert result.model["model"] == "line_based_radial_v1"
    assert result.model["k1"] == pytest.approx(reference_model["k1"], abs=0.12)
    assert result.mean_line_error < 0.75

    corrected_first_line = undistort_points_line_based(
        distorted_lines[0],
        model=result.model,
        image_width=image_width,
        image_height=image_height,
    )
    assert _line_rms_error(corrected_first_line) < 0.8
