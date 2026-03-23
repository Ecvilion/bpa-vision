"""Geometric primitives — homography application and point undistortion.

Ported from vision_box.services.brain.geometry — only the functions needed
for calibration and projection. No external deps beyond OpenCV/NumPy.
"""

from __future__ import annotations

import cv2  # type: ignore
import numpy as np


def apply_homography(point: tuple[float, float], matrix: list[list[float]]) -> tuple[float, float]:
    """Apply 3x3 homography matrix to a 2D point."""
    x, y = point
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        return point
    hx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    hy = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    hz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if hz == 0:
        return point
    return (hx / hz, hy / hz)


def undistort_point(
    point: tuple[float, float],
    intrinsic_matrix: list[list[float]] | None,
    dist_coeffs: list[float] | None,
) -> tuple[float, float]:
    """Undistort a single pixel-space point using camera intrinsics."""
    if not intrinsic_matrix or not dist_coeffs:
        return point
    camera_matrix = np.array(intrinsic_matrix, dtype=np.float64)
    distortion = np.array(dist_coeffs, dtype=np.float64)
    pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, camera_matrix, distortion, P=camera_matrix)
    return (float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1]))


def validate_homography_matrix(matrix: list[list[float]], label: str = "") -> None:
    """Validate that matrix is a valid 3x3 homography."""
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError(f"Homography {label} must be 3x3")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"Homography {label} must be 3x3")
        for v in row:
            if not isinstance(v, (int, float)):
                raise ValueError(f"Homography {label} contains non-numeric value")
    arr = np.array(matrix, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Homography {label} contains non-finite values")
    if abs(np.linalg.det(arr)) < 1e-10:
        raise ValueError(f"Homography {label} is singular")


__all__ = [
    "apply_homography",
    "undistort_point",
    "validate_homography_matrix",
]
