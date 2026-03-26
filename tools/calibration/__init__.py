"""Camera calibration tools — distortion correction and homography."""

from .distortion import (
    ChessboardCalibrationResult,
    calibrate_camera_from_chessboard_images,
    decode_image_bytes,
    undistort_image,
    undistort_points,
)
from .homography import compute_homography, reproject_points
from .geometry import apply_homography, undistort_point, validate_homography_matrix

__all__ = [
    "ChessboardCalibrationResult",
    "calibrate_camera_from_chessboard_images",
    "compute_homography",
    "decode_image_bytes",
    "reproject_points",
    "undistort_image",
    "undistort_points",
    "apply_homography",
    "undistort_point",
    "validate_homography_matrix",
]
