"""Camera calibration tools — distortion correction and homography."""

from .distortion import (
    ChessboardCalibrationResult,
    calibrate_camera_from_chessboard_images,
    decode_image_bytes,
    undistort_image,
    undistort_points,
)
from .homography import compute_homography, reproject_points
from .line_based import (
    compute_line_based_distortion,
    distort_points_line_based,
    undistort_image_line_based,
    undistort_points_line_based,
)
from .geometry import apply_homography, undistort_point, validate_homography_matrix

__all__ = [
    "ChessboardCalibrationResult",
    "calibrate_camera_from_chessboard_images",
    "compute_line_based_distortion",
    "compute_homography",
    "decode_image_bytes",
    "distort_points_line_based",
    "reproject_points",
    "undistort_image",
    "undistort_image_line_based",
    "undistort_points",
    "undistort_points_line_based",
    "apply_homography",
    "undistort_point",
    "validate_homography_matrix",
]
