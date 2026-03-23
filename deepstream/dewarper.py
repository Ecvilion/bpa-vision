"""Generate nvdewarper config files from OpenCV calibration parameters.

nvdewarper performs GPU-accelerated lens distortion correction.
projection-type=3 corresponds to perspective distortion model using
the same k1,k2,k3,p1,p2 coefficients as OpenCV.

Ported from vision_box.services.perception.deepstream_dewarper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DewarperConfig:
    camera_id: str
    config_file: str
    width: int
    height: int


def _sanitize(camera_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(camera_id or "").strip())
    return token.strip("._-") or "camera"


def opencv_to_dewarper_coeffs(distortion_coefficients: list[float]) -> tuple[float, ...]:
    """Convert OpenCV distortion [k1,k2,p1,p2,k3] to nvdewarper order [k1,k2,k3,p1,p2]."""
    v = [float(x) for x in distortion_coefficients]
    k1 = v[0] if len(v) > 0 else 0.0
    k2 = v[1] if len(v) > 1 else 0.0
    p1 = v[2] if len(v) > 2 else 0.0
    p2 = v[3] if len(v) > 3 else 0.0
    k3 = v[4] if len(v) > 4 else 0.0
    return (k1, k2, k3, p1, p2)


def render_dewarper_config(
    *,
    camera_id: str,
    intrinsic_matrix: list[list[float]],
    distortion_coefficients: list[float],
    width: int,
    height: int,
    output_dir: str | Path,
) -> DewarperConfig:
    """Write nvdewarper .txt config and return metadata."""
    if len(intrinsic_matrix) != 3 or any(len(row) != 3 for row in intrinsic_matrix):
        raise ValueError("intrinsic_matrix must be 3x3")

    w = max(1, int(width))
    h = max(1, int(height))
    fx = float(intrinsic_matrix[0][0])
    fy = float(intrinsic_matrix[1][1])
    cx = float(intrinsic_matrix[0][2])
    cy = float(intrinsic_matrix[1][2])
    coeffs = opencv_to_dewarper_coeffs(distortion_coefficients)

    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_sanitize(camera_id)}.dewarper.txt"
    path = base_dir / filename

    text = "\n".join([
        "[property]",
        f"output-width={w}",
        f"output-height={h}",
        "num-batch-buffers=1",
        "",
        "[surface0]",
        "projection-type=3",
        f"width={w}",
        f"height={h}",
        f"focal-length={fx:.9g};{fy:.9g}",
        f"src-x0={cx:.9g}",
        f"src-y0={cy:.9g}",
        "top-angle=0",
        "bottom-angle=0",
        "pitch=0",
        "yaw=0",
        "roll=0",
        "distortion=" + ";".join(f"{v:.9g}" for v in coeffs),
        "",
    ])
    path.write_text(text, encoding="utf-8")

    return DewarperConfig(
        camera_id=str(camera_id),
        config_file=str(path),
        width=w,
        height=h,
    )
