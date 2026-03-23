"""Chessboard-based camera calibration for intrinsics and distortion coefficients.

Ported from vision_box.tools.calibration.distortion — standalone, no external deps
beyond OpenCV and NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2  # type: ignore
import numpy as np


@dataclass(frozen=True)
class ChessboardCalibrationResult:
    intrinsic_matrix: list[list[float]]
    distortion_coefficients: list[float]
    image_width: int
    image_height: int
    rms_reprojection_error: float
    mean_reprojection_error: float
    valid_image_count: int
    rejected_image_count: int
    detected_files: tuple[str, ...]
    rejected_files: tuple[str, ...]


def _validate_pattern_size(pattern_cols: int, pattern_rows: int) -> tuple[int, int]:
    cols = int(pattern_cols)
    rows = int(pattern_rows)
    if cols < 2 or rows < 2:
        raise ValueError("pattern_cols and pattern_rows must be >= 2")
    return (cols, rows)


def _validate_square_size(square_size: float) -> float:
    size = float(square_size)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError("square_size must be a finite positive number")
    return size


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def _build_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size)
    return objp


def _find_chessboard_corners(image: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = 0
        flags |= int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
        flags |= int(getattr(cv2, "CALIB_CB_ACCURACY", 0))
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if ok and corners is not None:
            return np.asarray(corners, dtype=np.float32)

    ok, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=(
            int(getattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 1))
            | int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 2))
            | int(getattr(cv2, "CALIB_CB_FAST_CHECK", 8))
        ),
    )
    if not ok or corners is None:
        return None
    criteria = (
        int(cv2.TERM_CRITERIA_EPS) | int(cv2.TERM_CRITERIA_MAX_ITER),
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(refined, dtype=np.float32)


def calibrate_camera_from_chessboard_images(
    images: Iterable[tuple[str, np.ndarray]],
    *,
    pattern_cols: int,
    pattern_rows: int,
    square_size: float = 1.0,
) -> ChessboardCalibrationResult:
    pattern_size = _validate_pattern_size(pattern_cols, pattern_rows)
    square = _validate_square_size(square_size)
    object_template = _build_object_points(pattern_size, square)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    detected_files: list[str] = []
    rejected_files: list[str] = []
    image_size: tuple[int, int] | None = None

    for filename, image in images:
        if image is None or image.size == 0:
            rejected_files.append(str(filename))
            continue
        height, width = image.shape[:2]
        current_size = (int(width), int(height))
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            raise ValueError(
                "All chessboard images must have the same size. "
                f"Expected {image_size[0]}x{image_size[1]}, got {width}x{height} for {filename}."
            )
        corners = _find_chessboard_corners(image, pattern_size)
        if corners is None:
            rejected_files.append(str(filename))
            continue
        object_points.append(object_template.copy())
        image_points.append(corners.reshape(-1, 1, 2))
        detected_files.append(str(filename))

    if image_size is None:
        raise ValueError("No images supplied")
    if len(image_points) < 3:
        raise ValueError("Need at least 3 successful chessboard detections to calibrate the camera")

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    if camera_matrix is None or distortion is None:
        raise ValueError("OpenCV camera calibration failed")

    per_view_errors: list[float] = []
    for obj_pts, img_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs, strict=True):
        reprojected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, distortion)
        diff = np.asarray(img_pts, dtype=np.float64) - np.asarray(reprojected, dtype=np.float64)
        per_view_errors.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=2)))))

    distortion_vector = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if distortion_vector.size < 5:
        distortion_vector = np.pad(distortion_vector, (0, 5 - distortion_vector.size), mode="constant")

    return ChessboardCalibrationResult(
        intrinsic_matrix=[[float(v) for v in row] for row in np.asarray(camera_matrix, dtype=np.float64).tolist()],
        distortion_coefficients=[float(v) for v in distortion_vector.tolist()],
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        rms_reprojection_error=float(rms),
        mean_reprojection_error=float(sum(per_view_errors) / len(per_view_errors)),
        valid_image_count=len(detected_files),
        rejected_image_count=len(rejected_files),
        detected_files=tuple(detected_files),
        rejected_files=tuple(rejected_files),
    )


def undistort_image(
    image: np.ndarray,
    *,
    intrinsic_matrix: list[list[float]],
    distortion_coefficients: list[float],
    alpha: float = 0.0,
) -> tuple[np.ndarray, list[list[float]], tuple[int, int, int, int]]:
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    camera_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsic_matrix must be 3x3")
    distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
    height, width = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (int(width), int(height)),
        float(alpha),
        (int(width), int(height)),
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (int(width), int(height)),
        cv2.CV_16SC2,
    )
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    x, y, roi_w, roi_h = roi
    return (
        undistorted,
        [[float(v) for v in row] for row in np.asarray(new_camera_matrix, dtype=np.float64).tolist()],
        (int(x), int(y), int(roi_w), int(roi_h)),
    )


__all__ = [
    "ChessboardCalibrationResult",
    "calibrate_camera_from_chessboard_images",
    "decode_image_bytes",
    "undistort_image",
]
