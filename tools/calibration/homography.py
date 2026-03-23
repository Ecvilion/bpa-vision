"""Homography computation from point pairs using OpenCV.

Ported from vision_box.tools.calibration.homography — self-contained,
uses local geometry module instead of vision_box imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2  # type: ignore
import numpy as np

from .geometry import apply_homography, validate_homography_matrix


@dataclass(frozen=True)
class HomographyResult:
    """Computed homography with diagnostics for UI and persistence."""
    matrix: list[list[float]]
    reprojection_error: float
    inliers: int
    estimator: str
    reprojected_points: list[tuple[float, float]]
    inlier_mask: list[bool]
    inlier_reprojection_error: float
    median_reprojection_error: float
    max_reprojection_error: float
    source_coverage: float
    destination_coverage: float


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
    result = _compute_homography_impl(src_points, dst_points)
    return result.matrix, result.reprojection_error


def _coverage_ratio(points: np.ndarray) -> float:
    """Estimate how well points cover their bounding box.

    0.0 means degenerate / highly clustered. 1.0 means points span the full box.
    """
    if points.shape[0] < 3:
        return 0.0
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())
    bbox_area = max(0.0, (x_max - x_min) * (y_max - y_min))
    if bbox_area <= 0.0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    return max(0.0, min(1.0, hull_area / bbox_area))


def _compute_homography_impl(
    src_points: list[tuple[float, float]] | list[list[float]],
    dst_points: list[tuple[float, float]] | list[list[float]],
) -> HomographyResult:
    src = _to_point_array(src_points)
    dst = _to_point_array(dst_points)
    if src.shape[0] != dst.shape[0]:
        raise ValueError("src_points and dst_points must have equal length")
    if src.shape[0] < 4:
        raise ValueError("At least 4 point pairs are required")
    if np.unique(src, axis=0).shape[0] < 4:
        raise ValueError("At least 4 unique source points are required")
    if np.unique(dst, axis=0).shape[0] < 4:
        raise ValueError("At least 4 unique destination points are required")

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

    mask_values = (
        np.asarray(inlier_mask, dtype=np.int32).reshape(-1).astype(bool).tolist()
        if inlier_mask is not None
        else [True] * len(projected)
    )
    inlier_errors = [err for err, ok in zip(errors, mask_values) if ok]

    return HomographyResult(
        matrix=matrix,
        reprojection_error=reprojection_error,
        inliers=inliers,
        estimator=estimator,
        reprojected_points=projected,
        inlier_mask=mask_values,
        inlier_reprojection_error=(
            float(sum(inlier_errors) / len(inlier_errors)) if inlier_errors else 0.0
        ),
        median_reprojection_error=(
            float(np.median(np.asarray(errors, dtype=np.float64))) if errors else 0.0
        ),
        max_reprojection_error=max(errors) if errors else 0.0,
        source_coverage=_coverage_ratio(src),
        destination_coverage=_coverage_ratio(dst),
    )


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
