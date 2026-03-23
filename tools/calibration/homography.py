"""Homography computation from point pairs using OpenCV.

Ported from vision_box.tools.calibration.homography — self-contained,
uses local geometry module instead of vision_box imports.
"""

from __future__ import annotations

import math
from typing import Iterable

import cv2  # type: ignore
import numpy as np

from .geometry import apply_homography, validate_homography_matrix


def _to_point_array(points: Iterable[tuple[float, float] | list[float]]) -> np.ndarray:
    out: list[tuple[float, float]] = []
    for raw in points:
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError("Point must be pair [x, y]")
        x = float(raw[0])
        y = float(raw[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Point coordinates must be finite")
        out.append((x, y))
    if not out:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def compute_homography(
    src_points: list[tuple[float, float]] | list[list[float]],
    dst_points: list[tuple[float, float]] | list[list[float]],
) -> tuple[list[list[float]], float]:
    """Compute 3x3 homography from >=4 point pairs.

    Returns:
        (matrix_3x3_as_nested_lists, mean_reprojection_error_px)
    """
    matrix, reprojection_error, _inliers, _estimator = _compute_homography_impl(
        src_points, dst_points
    )
    return matrix, reprojection_error


def _compute_homography_impl(
    src_points: list[tuple[float, float]] | list[list[float]],
    dst_points: list[tuple[float, float]] | list[list[float]],
) -> tuple[list[list[float]], float, int, str]:
    src = _to_point_array(src_points)
    dst = _to_point_array(dst_points)
    if src.shape[0] != dst.shape[0]:
        raise ValueError("src_points and dst_points must have equal length")
    if src.shape[0] < 4:
        raise ValueError("At least 4 point pairs are required")

    h_matrix = None
    inlier_mask = None
    estimator = "RANSAC"
    if hasattr(cv2, "UsacParams") and hasattr(cv2, "SCORE_METHOD_MAGSAC"):
        params = cv2.UsacParams()
        params.confidence = 0.999
        params.maxIterations = 10_000
        params.threshold = 5.0
        params.sampler = int(getattr(cv2, "SAMPLING_UNIFORM", 0))
        params.score = int(getattr(cv2, "SCORE_METHOD_MAGSAC", 2))
        params.loMethod = int(getattr(cv2, "LOCAL_OPTIM_SIGMA", 4))
        params.loIterations = 10
        params.neighborsSearch = int(getattr(cv2, "NEIGH_GRID", 1))
        try:
            h_matrix, inlier_mask = cv2.findHomography(
                src.astype(np.float32),
                dst.astype(np.float32),
                params=params,
            )
            if h_matrix is not None:
                estimator = "USAC_MAGSAC"
        except Exception:
            h_matrix = None
            inlier_mask = None
            estimator = "RANSAC"
    if h_matrix is None:
        h_matrix, inlier_mask = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
    if h_matrix is None:
        raise ValueError("Failed to compute homography")

    inliers = 0
    if inlier_mask is not None:
        inliers = int(np.asarray(inlier_mask, dtype=np.int32).reshape(-1).sum())
    if inliers < 4:
        raise ValueError("Homography has fewer than 4 inliers")

    matrix = [[float(v) for v in row] for row in np.asarray(h_matrix, dtype=np.float64).tolist()]
    validate_homography_matrix(matrix, "calibration")

    projected = reproject_points(
        [(float(p[0]), float(p[1])) for p in src.tolist()], matrix
    )
    errors: list[float] = []
    for idx, proj in enumerate(projected):
        px = float(proj[0])
        py = float(proj[1])
        dx = px - float(dst[idx][0])
        dy = py - float(dst[idx][1])
        errors.append(math.sqrt(dx * dx + dy * dy))
    reprojection_error = float(sum(errors) / len(errors)) if errors else 0.0
    return matrix, reprojection_error, inliers, estimator


def reproject_points(
    src_points: list[tuple[float, float]] | list[list[float]],
    matrix: list[list[float]],
) -> list[tuple[float, float]]:
    """Apply homography matrix to all source points."""
    validate_homography_matrix(matrix, "calibration")
    points = _to_point_array(src_points)
    out: list[tuple[float, float]] = []
    for x, y in points.tolist():
        px, py = apply_homography((float(x), float(y)), matrix)
        out.append((float(px), float(py)))
    return out


__all__ = [
    "compute_homography",
    "reproject_points",
]
