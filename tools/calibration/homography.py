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

DEFAULT_RANSAC_REPROJ_THRESHOLD = 5.0
DEFAULT_CONFIDENCE = 0.999
DEFAULT_MAX_ITERATIONS = 10_000
DEFAULT_HOMOGRAPHY_METHOD = "auto"
HOMOGRAPHY_METHODS = {"auto", "all_points", "ransac", "usac_magsac"}


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
    *,
    homography_method: str = DEFAULT_HOMOGRAPHY_METHOD,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[list[list[float]], float]:
    """Compute 3x3 homography from >=4 point pairs.

    Returns:
        (matrix_3x3_as_nested_lists, mean_reprojection_error_px)
    """
    result = _compute_homography_impl(
        src_points,
        dst_points,
        homography_method=homography_method,
        ransac_reproj_threshold=ransac_reproj_threshold,
        confidence=confidence,
        max_iterations=max_iterations,
    )
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
    *,
    homography_method: str = DEFAULT_HOMOGRAPHY_METHOD,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
    homography_method = str(homography_method or DEFAULT_HOMOGRAPHY_METHOD).strip().lower()
    if homography_method not in HOMOGRAPHY_METHODS:
        raise ValueError(f"Unsupported homography_method: {homography_method}")
    if not math.isfinite(ransac_reproj_threshold) or ransac_reproj_threshold <= 0.0:
        raise ValueError("ransac_reproj_threshold must be positive")
    if not math.isfinite(confidence) or confidence <= 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in (0, 1]")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    h_matrix = None
    inlier_mask = None
    estimator = "RANSAC"
    has_usac_magsac = hasattr(cv2, "UsacParams") and hasattr(cv2, "SCORE_METHOD_MAGSAC")

    if homography_method == "all_points":
        h_matrix, inlier_mask = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=0,
        )
        estimator = "ALL_POINTS"
    elif homography_method == "usac_magsac":
        if not has_usac_magsac:
            raise ValueError("USAC_MAGSAC is not available in this OpenCV build")
        params = cv2.UsacParams()
        params.confidence = float(confidence)
        params.maxIterations = int(max_iterations)
        params.threshold = float(ransac_reproj_threshold)
        params.sampler = int(getattr(cv2, "SAMPLING_UNIFORM", 0))
        params.score = int(getattr(cv2, "SCORE_METHOD_MAGSAC", 2))
        params.loMethod = int(getattr(cv2, "LOCAL_OPTIM_SIGMA", 4))
        params.loIterations = 10
        params.neighborsSearch = int(getattr(cv2, "NEIGH_GRID", 1))
        h_matrix, inlier_mask = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            params=params,
        )
        estimator = "USAC_MAGSAC"
    elif homography_method == "ransac":
        h_matrix, inlier_mask = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_threshold),
            maxIters=int(max_iterations),
            confidence=float(confidence),
        )
        estimator = "RANSAC"
    elif has_usac_magsac:
        params = cv2.UsacParams()
        params.confidence = float(confidence)
        params.maxIterations = int(max_iterations)
        params.threshold = float(ransac_reproj_threshold)
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
    if h_matrix is None and homography_method == "auto":
        h_matrix, inlier_mask = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_threshold),
            maxIters=int(max_iterations),
            confidence=float(confidence),
        )
        estimator = "RANSAC"
    if h_matrix is None:
        raise ValueError("Failed to compute homography")

    if homography_method == "all_points":
        inlier_mask = np.ones((src.shape[0], 1), dtype=np.uint8)
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
