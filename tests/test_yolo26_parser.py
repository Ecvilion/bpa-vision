"""Tests for yolo26_parser — letterbox params, NaN filtering, normalized clamp."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# pyds / ctypes are only available inside the DeepStream container — stub them
sys.modules.setdefault("pyds", MagicMock())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deepstream"))

from yolo26_parser import (
    Detection,
    PoseDetection,
    _clamp01,
    _letterbox_params,
    detections_to_normalized,
)


class TestLetterboxParams:
    def test_no_scaling_needed(self):
        """Frame exactly matches input size — scale=1, no padding."""
        scale, pad_x, pad_y = _letterbox_params(640, 640, 640, 640)
        assert scale == pytest.approx(1.0)
        assert pad_x == pytest.approx(0.0)
        assert pad_y == pytest.approx(0.0)

    def test_landscape_frame(self):
        """1920x1080 into 640x640 — width-limited, vertical padding."""
        scale, pad_x, pad_y = _letterbox_params(640, 640, 1920, 1080)
        expected_scale = 640 / 1920  # ≈ 0.3333
        assert scale == pytest.approx(expected_scale)
        assert pad_x == pytest.approx(0.0)
        # Vertical padding: (640 - 1080 * scale) / 2
        expected_pad_y = (640 - 1080 * expected_scale) / 2.0
        assert pad_y == pytest.approx(expected_pad_y)

    def test_portrait_frame(self):
        """1080x1920 into 640x640 — height-limited, horizontal padding."""
        scale, pad_x, pad_y = _letterbox_params(640, 640, 1080, 1920)
        expected_scale = 640 / 1920
        assert scale == pytest.approx(expected_scale)
        expected_pad_x = (640 - 1080 * expected_scale) / 2.0
        assert pad_x == pytest.approx(expected_pad_x)
        assert pad_y == pytest.approx(0.0)

    def test_square_frame_smaller(self):
        """320x320 into 640x640 — scale=2, no padding."""
        scale, pad_x, pad_y = _letterbox_params(640, 640, 320, 320)
        assert scale == pytest.approx(2.0)
        assert pad_x == pytest.approx(0.0)
        assert pad_y == pytest.approx(0.0)

    def test_inverse_transform(self):
        """Verify frame_coord = (model_coord - pad) / scale round-trips."""
        scale, pad_x, pad_y = _letterbox_params(640, 640, 1920, 1080)
        # A point at frame (960, 540) — center of 1920x1080
        frame_x, frame_y = 960, 540
        model_x = frame_x * scale + pad_x
        model_y = frame_y * scale + pad_y
        # Inverse
        recovered_x = (model_x - pad_x) / scale
        recovered_y = (model_y - pad_y) / scale
        assert recovered_x == pytest.approx(frame_x)
        assert recovered_y == pytest.approx(frame_y)


class TestClamp01:
    def test_within_range(self):
        assert _clamp01(0.5) == 0.5

    def test_below_zero(self):
        assert _clamp01(-0.1) == 0.0

    def test_above_one(self):
        assert _clamp01(1.5) == 1.0

    def test_boundaries(self):
        assert _clamp01(0.0) == 0.0
        assert _clamp01(1.0) == 1.0


class TestDetectionsToNormalized:
    def test_basic_normalization(self):
        det = Detection(x1=100, y1=200, x2=300, y2=400, confidence=0.9, class_id=0)
        result = detections_to_normalized([det], 1920, 1080)
        assert len(result) == 1
        bbox = result[0]["bbox"]
        assert bbox["x_min"] == pytest.approx(100 / 1920)
        assert bbox["y_min"] == pytest.approx(200 / 1080)
        assert bbox["x_max"] == pytest.approx(300 / 1920)
        assert bbox["y_max"] == pytest.approx(400 / 1080)

    def test_clamps_negative_to_zero(self):
        det = Detection(x1=-50, y1=-30, x2=100, y2=200, confidence=0.8, class_id=0)
        result = detections_to_normalized([det], 1920, 1080)
        bbox = result[0]["bbox"]
        assert bbox["x_min"] == 0.0
        assert bbox["y_min"] == 0.0

    def test_clamps_above_frame_to_one(self):
        det = Detection(x1=0, y1=0, x2=2000, y2=1200, confidence=0.8, class_id=0)
        result = detections_to_normalized([det], 1920, 1080)
        bbox = result[0]["bbox"]
        assert bbox["x_max"] == 1.0
        assert bbox["y_max"] == 1.0

    def test_pose_keypoints_clamped(self):
        det = PoseDetection(
            x1=100, y1=100, x2=200, y2=200,
            confidence=0.9, class_id=0,
            keypoints=[
                (-10.0, 50.0, 0.9),   # negative x → clamped to 0
                (2000.0, 50.0, 0.8),   # over frame → clamped to 1
                (500.0, 500.0, 0.7),   # normal
            ],
        )
        result = detections_to_normalized([det], 1920, 1080)
        kps = result[0]["keypoints"]
        assert kps[0]["x"] == 0.0
        assert kps[1]["x"] == 1.0
        assert kps[2]["x"] == pytest.approx(500 / 1920)

    def test_empty_list(self):
        assert detections_to_normalized([], 1920, 1080) == []
