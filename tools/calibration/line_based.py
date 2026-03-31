"""Experimental line-based distortion estimation and correction.

This implements a simple plumb-line style v1 estimator:
- user provides multiple polylines that should be straight in the scene
- we fit a single radial distortion coefficient k1 around a fixed center
- image correction uses a forward radial warp from undistorted output pixels

The model is intentionally simple and marked experimental. It is useful for
quick comparison against board-based calibration, not as a drop-in replacement
for full camera calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import cv2  # type: ignore
import numpy as np


@dataclass(frozen=True)
class LineBasedDistortionResult:
    model: dict[str, Any]
    mean_line_error: float
    max_line_error: float
    line_errors: list[float]
    corrected_lines: list[list[tuple[float, float]]]
    line_count: int
    total_points: int


def _as_lines(lines: Iterable[Iterable[Iterable[float]]]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    total_points = 0
    for line_idx, line in enumerate(lines):
        points: list[tuple[float, float]] = []
        for point_idx, raw in enumerate(line):
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                raise ValueError(f"lines[{line_idx}][{point_idx}] must be [x, y]")
            x = float(raw[0])
            y = float(raw[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"lines[{line_idx}][{point_idx}] must be finite")
            points.append((x, y))
        if len(points) < 3:
            raise ValueError(f"lines[{line_idx}] must contain at least 3 points")
        out.append(np.asarray(points, dtype=np.float64))
        total_points += len(points)
    if len(out) < 2:
        raise ValueError("At least 2 lines are required for line-based fitting")
    if total_points < 8:
        raise ValueError("At least 8 total points are required for line-based fitting")
    return out


def _as_point_array(points: Iterable[tuple[float, float] | list[float]]) -> np.ndarray:
    out: list[tuple[float, float]] = []
    for idx, raw in enumerate(points):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError(f"points[{idx}] must be [x, y]")
        x = float(raw[0])
        y = float(raw[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"points[{idx}] must be finite")
        out.append((x, y))
    if not out:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _resolve_center(
    *,
    image_width: int,
    image_height: int,
    principal_point: list[float] | tuple[float, float] | None = None,
) -> tuple[float, float]:
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if principal_point is None:
        return (width / 2.0, height / 2.0)
    if not isinstance(principal_point, (tuple, list)) or len(principal_point) != 2:
        raise ValueError("principal_point must be [x, y]")
    cx = float(principal_point[0])
    cy = float(principal_point[1])
    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError("principal_point must be finite")
    return (cx, cy)


def _normalized_center(model: dict[str, Any], *, image_width: int, image_height: int) -> tuple[float, float]:
    center_norm = model.get("center_norm")
    if isinstance(center_norm, list) and len(center_norm) == 2:
        return (
            float(center_norm[0]) * float(image_width),
            float(center_norm[1]) * float(image_height),
        )
    center = model.get("center")
    if isinstance(center, list) and len(center) == 2:
        src_width = float(model.get("source_image_width") or image_width)
        src_height = float(model.get("source_image_height") or image_height)
        return (
            float(center[0]) * float(image_width) / max(1.0, src_width),
            float(center[1]) * float(image_height) / max(1.0, src_height),
        )
    return (float(image_width) / 2.0, float(image_height) / 2.0)


def _undistort_points_array(points: np.ndarray, *, k1: float, cx: float, cy: float, scale: float) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    distorted = np.asarray(points, dtype=np.float64)
    xd = (distorted[:, 0] - cx) / scale
    yd = (distorted[:, 1] - cy) / scale
    xu = xd.copy()
    yu = yd.copy()
    for _ in range(10):
        r2 = xu * xu + yu * yu
        factor = 1.0 + float(k1) * r2
        factor = np.where(np.abs(factor) < 1e-8, 1e-8, factor)
        xu = xd / factor
        yu = yd / factor
    out = np.empty_like(distorted, dtype=np.float64)
    out[:, 0] = xu * scale + cx
    out[:, 1] = yu * scale + cy
    return out


def undistort_points_line_based(
    points: Iterable[tuple[float, float] | list[float]],
    *,
    model: dict[str, Any],
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    k1 = float(model.get("k1", 0.0))
    cx, cy = _normalized_center(model, image_width=image_width, image_height=image_height)
    scale = float(max(int(image_width), int(image_height)))
    array = _as_point_array(points)
    corrected = _undistort_points_array(array, k1=k1, cx=cx, cy=cy, scale=scale)
    return [(float(point[0]), float(point[1])) for point in corrected.tolist()]


def distort_points_line_based(
    points: Iterable[tuple[float, float] | list[float]],
    *,
    model: dict[str, Any],
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    k1 = float(model.get("k1", 0.0))
    cx, cy = _normalized_center(model, image_width=image_width, image_height=image_height)
    scale = float(max(int(image_width), int(image_height)))
    array = _as_point_array(points)
    xu = (array[:, 0] - cx) / scale
    yu = (array[:, 1] - cy) / scale
    r2 = xu * xu + yu * yu
    factor = 1.0 + float(k1) * r2
    out = np.empty_like(array, dtype=np.float64)
    out[:, 0] = xu * factor * scale + cx
    out[:, 1] = yu * factor * scale + cy
    return [(float(point[0]), float(point[1])) for point in out.tolist()]


def undistort_image_line_based(image: np.ndarray, *, model: dict[str, Any]) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    height, width = image.shape[:2]
    cx, cy = _normalized_center(model, image_width=width, image_height=height)
    scale = float(max(width, height))
    k1 = float(model.get("k1", 0.0))

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    xu = (grid_x - float(cx)) / float(scale)
    yu = (grid_y - float(cy)) / float(scale)
    r2 = xu * xu + yu * yu
    factor = 1.0 + float(k1) * r2
    map_x = xu * factor * float(scale) + float(cx)
    map_y = yu * factor * float(scale) + float(cy)
    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _line_rms_error(points: np.ndarray) -> float:
    centered = np.asarray(points, dtype=np.float64)
    if centered.shape[0] < 2:
        return float("inf")
    centroid = centered.mean(axis=0, keepdims=True)
    centered = centered - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    distances = centered @ normal
    return float(np.sqrt(np.mean(distances * distances)))


def _evaluate_k1(lines: list[np.ndarray], *, k1: float, cx: float, cy: float, scale: float) -> tuple[float, list[float]]:
    errors: list[float] = []
    weighted_sum = 0.0
    total_points = 0
    for line in lines:
        corrected = _undistort_points_array(line, k1=k1, cx=cx, cy=cy, scale=scale)
        err = _line_rms_error(corrected)
        errors.append(err)
        weighted_sum += float(err * err) * float(corrected.shape[0])
        total_points += int(corrected.shape[0])
    loss = weighted_sum / max(1, total_points)
    return (float(loss), errors)


def compute_line_based_distortion(
    lines: Iterable[Iterable[Iterable[float]]],
    *,
    image_width: int,
    image_height: int,
    principal_point: list[float] | tuple[float, float] | None = None,
) -> LineBasedDistortionResult:
    line_arrays = _as_lines(lines)
    width = int(image_width)
    height = int(image_height)
    cx, cy = _resolve_center(
        image_width=width,
        image_height=height,
        principal_point=principal_point,
    )
    scale = float(max(width, height))

    best_k1 = 0.0
    best_loss, best_errors = _evaluate_k1(line_arrays, k1=best_k1, cx=cx, cy=cy, scale=scale)
    span = 4.0
    for samples in (161, 161, 121, 121):
        grid = np.linspace(best_k1 - span, best_k1 + span, samples, dtype=np.float64)
        for candidate in grid.tolist():
            loss, errors = _evaluate_k1(line_arrays, k1=float(candidate), cx=cx, cy=cy, scale=scale)
            if loss < best_loss:
                best_loss = loss
                best_k1 = float(candidate)
                best_errors = errors
        span /= 6.0

    corrected_lines = [
        [
            (float(point[0]), float(point[1]))
            for point in _undistort_points_array(line, k1=best_k1, cx=cx, cy=cy, scale=scale).tolist()
        ]
        for line in line_arrays
    ]
    model = {
        "model": "line_based_radial_v1",
        "k1": float(best_k1),
        "center": [float(cx), float(cy)],
        "center_norm": [float(cx) / float(width), float(cy) / float(height)],
        "source_image_width": width,
        "source_image_height": height,
        "scale_mode": "max_dim",
    }
    total_points = sum(int(line.shape[0]) for line in line_arrays)
    return LineBasedDistortionResult(
        model=model,
        mean_line_error=float(sum(best_errors) / len(best_errors)) if best_errors else 0.0,
        max_line_error=max(best_errors) if best_errors else 0.0,
        line_errors=[float(value) for value in best_errors],
        corrected_lines=corrected_lines,
        line_count=len(line_arrays),
        total_points=total_points,
    )


__all__ = [
    "LineBasedDistortionResult",
    "compute_line_based_distortion",
    "distort_points_line_based",
    "undistort_image_line_based",
    "undistort_points_line_based",
]
