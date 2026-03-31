"""Calibration web tool — FastAPI app for camera distortion & homography.

Debug utility for camera setup. NOT a production dependency.
Run standalone: python -m tools.calibration --configs-dir configs

Ported from vision_box.tools.calibration.app — stripped of vision_box deps.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2  # type: ignore
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .distortion import (
    calibrate_camera_from_chessboard_images,
    decode_image_bytes,
    undistort_image,
    undistort_points,
)
from .homography import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HOMOGRAPHY_METHOD,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    _compute_homography_impl,
)
from .line_based import (
    compute_line_based_distortion,
    undistort_image_line_based,
    undistort_points_line_based,
)
from .geometry import validate_homography_matrix


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_identity_homography(matrix: Any, *, eps: float = 1e-6) -> bool:
    if not isinstance(matrix, list) or len(matrix) != 3:
        return True
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for i in range(3):
        row = matrix[i]
        if not isinstance(row, list) or len(row) != 3:
            return True
        for j in range(3):
            try:
                val = float(row[j])
            except Exception:
                return True
            if abs(val - identity[i][j]) > float(eps):
                return False
    return True


_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]{1,255}$")
_ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def sanitize_filename(filename: str) -> str:
    name = Path(str(filename)).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("Invalid filename")
    if not _SAFE_FILENAME.match(name):
        raise ValueError("Filename contains unsupported characters")
    ext = Path(name).suffix.lower()
    if ext not in _ALLOWED_IMAGE_EXT:
        raise ValueError(f"Unsupported image extension: {ext}")
    return name


@dataclass(frozen=True)
class CalibrationRuntime:
    calibration_dir: Path
    floor_plans_dir: Path
    camera_urls: dict[str, str]
    camera_urls_path: Path | None = None
    site_config_path: Path | None = None
    perception_config_path: Path | None = None
    camera_frame_width: int = 2560
    camera_frame_height: int = 1440

    def _cal_path(self, camera_id: str) -> Path:
        return self.calibration_dir / f"calibration_{camera_id}.yaml"

    def load_camera_calibration(self, camera_id: str) -> dict[str, Any]:
        path = self._cal_path(camera_id)
        if not path.exists():
            return {"camera_id": camera_id}
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            return {"camera_id": camera_id}
        return raw

    def save_camera_calibration(self, camera_id: str, data: dict[str, Any]) -> Path:
        path = self._cal_path(camera_id)
        # Backup if exists
        if path.exists():
            ts = _utcnow().strftime("%Y%m%dT%H%M%S")
            backup = path.with_name(f"{path.name}.bak.{ts}")
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        data["camera_id"] = camera_id
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    def calibration_camera_ids(self) -> list[str]:
        if not self.calibration_dir.exists():
            return []
        prefix = "calibration_"
        suffix = ".yaml"
        ids: set[str] = set()
        for item in self.calibration_dir.glob(f"{prefix}*{suffix}"):
            name = item.name
            if name.startswith(prefix) and name.endswith(suffix):
                ids.add(name[len(prefix):-len(suffix)])
        return sorted(ids)

    def all_camera_ids(self) -> list[str]:
        return sorted(set(self.camera_urls) | set(self.calibration_camera_ids()))

    def floor_plan_file(self, filename: str) -> Path:
        safe = sanitize_filename(filename)
        base = self.floor_plans_dir.resolve()
        candidate = (base / safe).resolve()
        candidate.relative_to(base)  # raises if traversal
        return candidate

    def list_floor_plans(self) -> list[str]:
        self.floor_plans_dir.mkdir(parents=True, exist_ok=True)
        return sorted(
            item.name
            for item in self.floor_plans_dir.iterdir()
            if item.is_file() and item.suffix.lower() in _ALLOWED_IMAGE_EXT
        )


class ComputeHomographyRequest(BaseModel):
    src_points: list[list[float]]
    dst_points: list[list[float]]
    homography_method: Literal["auto", "all_points", "ransac", "usac_magsac"] = DEFAULT_HOMOGRAPHY_METHOD
    ransac_reproj_threshold: float = Field(default=DEFAULT_RANSAC_REPROJ_THRESHOLD, gt=0.0)
    confidence: float = Field(default=DEFAULT_CONFIDENCE, gt=0.0, le=1.0)
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1)


class CalibrationPointPair(BaseModel):
    camera_point: list[float] | None = None
    plan_point: list[float] | None = None


class SaveCalibrationRequest(BaseModel):
    camera_id: str
    rtsp_url: str | None = None
    anchor_point: str | None = None
    camera_points_space: Literal["raw", "undistorted"] | None = None
    matrix: list[list[float]] | None = None
    matrix_ground: list[list[float]] | None = None
    matrix_hip: list[list[float]] | None = None
    matrix_head: list[list[float]] | None = None
    floor_plan_filename: str = ""
    frame_width: int | None = None
    frame_height: int | None = None
    intrinsic_matrix: list[list[float]] | None = None
    distortion_coefficients: list[float] | None = None
    distortion_correction_mode: str | None = None
    coverage_polygon: list[list[float]] | None = None
    point_pairs: list[CalibrationPointPair] | None = None
    line_constraints: list[list[list[float]]] | None = None
    line_based_distortion: dict[str, Any] | None = None
    line_based_stats: dict[str, Any] | None = None
    homography_stats: dict[str, Any] | None = None


def _camera_has_valid_homography(calibration: dict[str, Any]) -> bool:
    matrix = calibration.get("homography_ground") or calibration.get("homography_matrix")
    return not _is_identity_homography(matrix)


class CaptureFrameRequest(BaseModel):
    camera_id: str | None = None
    rtsp_url: str | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    intrinsic_matrix: list[list[float]] | None = None
    distortion_coefficients: list[float] | None = None
    distortion_correction_mode: str | None = None
    line_based_distortion: dict[str, Any] | None = None
    apply_undistort: bool = False
    undistort_alpha: float = 0.0


class UndistortCameraPointsRequest(BaseModel):
    points: list[list[float] | None]
    intrinsic_matrix: list[list[float]] | None = None
    distortion_coefficients: list[float] | None = None
    distortion_correction_mode: str | None = None
    line_based_distortion: dict[str, Any] | None = None
    image_width: int
    image_height: int
    alpha: float = 0.0


class ComputeLineBasedDistortionRequest(BaseModel):
    lines: list[list[list[float]]]
    image_width: int
    image_height: int
    principal_point: list[float] | None = None


def capture_rtsp_frame(rtsp_url: str, *, timeout_sec: float = 5.0) -> bytes:
    cap = None
    start = time.monotonic()
    try:
        cap = cv2.VideoCapture(str(rtsp_url), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(str(rtsp_url))
        if not cap.isOpened():
            raise RuntimeError("VideoCapture open failed")
        frame = None
        while time.monotonic() - start <= float(timeout_sec):
            ok, current = cap.read()
            if ok and current is not None:
                frame = np.asarray(current).copy()
                break
            time.sleep(0.05)
        if frame is None:
            raise TimeoutError(f"Timed out reading frame after {timeout_sec:.1f}s")
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode JPEG")
        return bytes(encoded.tobytes())
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _resize_jpeg_if_needed(jpeg: bytes, *, width: int, height: int) -> bytes:
    if width <= 0 or height <= 0:
        return jpeg
    raw = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jpeg
    src_h, src_w = frame.shape[:2]
    if src_w == width and src_h == height:
        return jpeg
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized)
    return bytes(encoded.tobytes()) if ok else jpeg


def _index_html() -> str:
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")


def create_calibration_app(runtime: CalibrationRuntime) -> FastAPI:
    app = FastAPI(title="BPA Vision Calibration Tool", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _index_html()

    @app.get("/api/config")
    async def config_meta() -> JSONResponse:
        return JSONResponse({
            "camera_frame_width": runtime.camera_frame_width,
            "camera_frame_height": runtime.camera_frame_height,
            "calibration_dir": str(runtime.calibration_dir),
            "floor_plans_dir": str(runtime.floor_plans_dir),
            "site_config_path": str(runtime.site_config_path) if runtime.site_config_path else "",
            "camera_urls_path": str(runtime.camera_urls_path) if runtime.camera_urls_path else "",
            "floor_plans_dir_path": str(runtime.floor_plans_dir),
            "perception_config_path": (
                str(runtime.perception_config_path) if runtime.perception_config_path else ""
            ),
        })

    @app.get("/api/cameras")
    async def cameras() -> JSONResponse:
        out: list[dict[str, Any]] = []
        for camera_id in runtime.all_camera_ids():
            cal = runtime.load_camera_calibration(camera_id)
            rtsp_url = runtime.camera_urls.get(camera_id) or cal.get("rtsp_url", "")
            matrix = cal.get("homography_ground") or cal.get("homography_matrix")
            out.append({
                "camera_id": camera_id,
                "has_homography": _camera_has_valid_homography(cal),
                "rtsp_url": rtsp_url,
                "anchor_point": cal.get("anchor_point", "bottom_center"),
                "camera_points_space": cal.get("camera_points_space", "raw"),
                "homography_matrix": cal.get("homography_matrix"),
                "homography_ground": cal.get("homography_ground") or matrix,
                "homography_hip": cal.get("homography_hip"),
                "homography_head": cal.get("homography_head"),
                "floor_plan_image": cal.get("floor_plan_image"),
                "frame_width": cal.get("frame_width"),
                "frame_height": cal.get("frame_height"),
                "intrinsic_matrix": cal.get("intrinsic_matrix"),
                "distortion_coefficients": cal.get("distortion_coefficients"),
                "distortion_correction_mode": cal.get("distortion_correction_mode", "none"),
                "coverage_polygon": cal.get("coverage_polygon", []),
                "point_pairs": cal.get("point_pairs", []),
                "line_constraints": cal.get("line_constraints", []),
                "line_based_distortion": cal.get("line_based_distortion"),
                "line_based_stats": cal.get("line_based_stats", {}),
                "homography_stats": cal.get("homography_stats", {}),
            })
        return JSONResponse({"cameras": sorted(out, key=lambda x: x["camera_id"])})

    @app.get("/api/floor-plans")
    async def floor_plans() -> JSONResponse:
        return JSONResponse({"floor_plans": runtime.list_floor_plans()})

    @app.get("/api/floor-plan/{filename}")
    async def floor_plan_image(filename: str) -> FileResponse:
        try:
            path = runtime.floor_plan_file(filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(path))

    @app.post("/api/upload-floor-plan")
    async def upload_floor_plan(file: UploadFile = File(...)) -> JSONResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename")
        try:
            safe_name = sanitize_filename(file.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        dest = runtime.floor_plans_dir / safe_name
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        dest.write_bytes(content)
        return JSONResponse({"filename": safe_name, "size": len(content)})

    @app.post("/api/camera/frame")
    async def capture_frame_endpoint(req: CaptureFrameRequest) -> Response:
        rtsp_url = req.rtsp_url
        if not rtsp_url and req.camera_id:
            rtsp_url = runtime.camera_urls.get(req.camera_id)
        if not rtsp_url:
            raise HTTPException(status_code=400, detail="No RTSP URL")
        try:
            jpeg = capture_rtsp_frame(rtsp_url)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        target_w = req.frame_width or runtime.camera_frame_width
        target_h = req.frame_height or runtime.camera_frame_height
        jpeg = _resize_jpeg_if_needed(jpeg, width=target_w, height=target_h)

        if (
            req.apply_undistort
            and req.distortion_correction_mode == "line_based_v1"
            and req.line_based_distortion
        ):
            raw = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is not None:
                corrected = undistort_image_line_based(
                    frame,
                    model=req.line_based_distortion,
                )
                ok, encoded = cv2.imencode(".jpg", corrected)
                if ok:
                    jpeg = bytes(encoded.tobytes())
        elif req.apply_undistort and req.intrinsic_matrix and req.distortion_coefficients:
            raw = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is not None:
                undistorted, _, _ = undistort_image(
                    frame,
                    intrinsic_matrix=req.intrinsic_matrix,
                    distortion_coefficients=req.distortion_coefficients,
                    alpha=req.undistort_alpha,
                )
                ok, encoded = cv2.imencode(".jpg", undistorted)
                if ok:
                    jpeg = bytes(encoded.tobytes())

        return Response(content=jpeg, media_type="image/jpeg")

    @app.post("/api/points/undistort")
    async def undistort_camera_points(req: UndistortCameraPointsRequest) -> JSONResponse:
        filtered_points: list[list[float]] = []
        filtered_indices: list[int] = []
        for idx, point in enumerate(req.points):
            if point is None:
                continue
            if len(point) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"points[{idx}] must be [x, y] or null",
                )
            filtered_points.append([float(point[0]), float(point[1])])
            filtered_indices.append(idx)
        try:
            if req.distortion_correction_mode == "line_based_v1" and req.line_based_distortion:
                mapped_points = undistort_points_line_based(
                    filtered_points,
                    model=req.line_based_distortion,
                    image_width=req.image_width,
                    image_height=req.image_height,
                )
                new_intrinsic_matrix = None
                roi = (0, 0, int(req.image_width), int(req.image_height))
            else:
                if req.intrinsic_matrix is None or req.distortion_coefficients is None:
                    raise ValueError(
                        "intrinsic_matrix and distortion_coefficients are required",
                    )
                mapped_points, new_intrinsic_matrix, roi = undistort_points(
                    filtered_points,
                    intrinsic_matrix=req.intrinsic_matrix,
                    distortion_coefficients=req.distortion_coefficients,
                    image_width=req.image_width,
                    image_height=req.image_height,
                    alpha=req.alpha,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        result_points: list[list[float] | None] = [None] * len(req.points)
        for idx, point in zip(filtered_indices, mapped_points, strict=True):
            result_points[idx] = [float(point[0]), float(point[1])]
        return JSONResponse({
            "points": result_points,
            "camera_points_space": "undistorted",
            "new_intrinsic_matrix": new_intrinsic_matrix,
            "roi": [int(v) for v in roi],
        })

    @app.post("/api/distortion/line-based")
    async def calibrate_line_based_distortion(req: ComputeLineBasedDistortionRequest) -> JSONResponse:
        try:
            result = compute_line_based_distortion(
                req.lines,
                image_width=req.image_width,
                image_height=req.image_height,
                principal_point=req.principal_point,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse({
            "model": result.model,
            "mean_line_error": result.mean_line_error,
            "max_line_error": result.max_line_error,
            "line_errors": result.line_errors,
            "corrected_lines": [
                [[float(point[0]), float(point[1])] for point in line]
                for line in result.corrected_lines
            ],
            "line_count": result.line_count,
            "total_points": result.total_points,
        })

    @app.post("/api/compute-homography")  # also aliased below
    async def compute_homography_endpoint(req: ComputeHomographyRequest) -> JSONResponse:
        try:
            result = _compute_homography_impl(
                req.src_points,
                req.dst_points,
                homography_method=req.homography_method,
                ransac_reproj_threshold=req.ransac_reproj_threshold,
                confidence=req.confidence,
                max_iterations=req.max_iterations,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse({
            "matrix": result.matrix,
            "reprojected_points": [
                [float(point[0]), float(point[1])]
                for point in result.reprojected_points
            ],
            "inlier_mask": list(result.inlier_mask),
            "reprojection_error": result.reprojection_error,
            "inlier_reprojection_error": result.inlier_reprojection_error,
            "median_reprojection_error": result.median_reprojection_error,
            "max_reprojection_error": result.max_reprojection_error,
            "source_coverage": result.source_coverage,
            "destination_coverage": result.destination_coverage,
            "inliers": result.inliers,
            "estimator": result.estimator,
            "homography_method": req.homography_method,
            "ransac_reproj_threshold": req.ransac_reproj_threshold,
            "confidence": req.confidence,
            "max_iterations": req.max_iterations,
        })

    @app.post("/api/save")
    async def save_calibration(req: SaveCalibrationRequest) -> JSONResponse:
        cam_id = req.camera_id.strip()
        if not cam_id:
            raise HTTPException(status_code=400, detail="camera_id required")

        cal = runtime.load_camera_calibration(cam_id)

        def _save_matrix(field_name: str, matrix: list[list[float]] | None) -> bool:
            if not matrix:
                return False
            validate_homography_matrix(matrix, f"{cam_id}:{field_name}")
            cal[field_name] = matrix
            return True

        saved_planes = {
            "base": _save_matrix("homography_matrix", req.matrix),
            "ground": _save_matrix("homography_ground", req.matrix_ground or req.matrix),
            "hip": _save_matrix("homography_hip", req.matrix_hip),
            "head": _save_matrix("homography_head", req.matrix_head),
        }
        if req.rtsp_url is not None:
            cal["rtsp_url"] = req.rtsp_url
        if req.anchor_point:
            cal["anchor_point"] = req.anchor_point
        if req.camera_points_space is not None:
            cal["camera_points_space"] = req.camera_points_space
        if req.floor_plan_filename:
            cal["floor_plan_image"] = req.floor_plan_filename
        if req.frame_width:
            cal["frame_width"] = req.frame_width
        if req.frame_height:
            cal["frame_height"] = req.frame_height
        if req.intrinsic_matrix:
            cal["intrinsic_matrix"] = req.intrinsic_matrix
        if req.distortion_coefficients is not None:
            cal["distortion_coefficients"] = req.distortion_coefficients
        if req.distortion_correction_mode:
            cal["distortion_correction_mode"] = req.distortion_correction_mode
        if req.coverage_polygon is not None:
            cal["coverage_polygon"] = req.coverage_polygon
        if req.point_pairs is not None:
            cal["point_pairs"] = [pair.model_dump() for pair in req.point_pairs]
        if req.line_constraints is not None:
            cal["line_constraints"] = req.line_constraints
        if req.line_based_distortion is not None:
            cal["line_based_distortion"] = req.line_based_distortion
        if req.line_based_stats is not None:
            cal["line_based_stats"] = req.line_based_stats
        if req.homography_stats is not None:
            cal["homography_stats"] = req.homography_stats

        path = runtime.save_camera_calibration(cam_id, cal)
        return JSONResponse({
            "ok": True,
            "path": str(path),
            "saved_at": _utcnow().isoformat(),
            "saved_planes": saved_planes,
        })

    @app.post("/api/distortion/calibrate")
    async def calibrate_distortion(
        pattern_cols: int = Form(...),
        pattern_rows: int = Form(...),
        square_size: float = Form(1.0),
        files: list[UploadFile] = File(...),
    ) -> JSONResponse:
        images = []
        for f in files:
            content = await f.read()
            try:
                img = decode_image_bytes(content)
                images.append((f.filename or "unknown", img))
            except ValueError:
                pass

        try:
            result = calibrate_camera_from_chessboard_images(
                images,
                pattern_cols=pattern_cols,
                pattern_rows=pattern_rows,
                square_size=square_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return JSONResponse({
            "intrinsic_matrix": result.intrinsic_matrix,
            "distortion_coefficients": result.distortion_coefficients,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "rms_reprojection_error": result.rms_reprojection_error,
            "mean_reprojection_error": result.mean_reprojection_error,
            "valid_image_count": result.valid_image_count,
            "rejected_image_count": result.rejected_image_count,
            "detected_files": list(result.detected_files),
            "rejected_files": list(result.rejected_files),
        })

    @app.get("/api/readiness")
    async def readiness() -> JSONResponse:
        camera_checks: list[dict[str, Any]] = []
        for camera_id in runtime.all_camera_ids():
            cal = runtime.load_camera_calibration(camera_id)
            has_homography = _camera_has_valid_homography(cal)
            has_floor_plan = bool(cal.get("floor_plan_image"))
            has_coverage = len(cal.get("coverage_polygon") or []) >= 3
            has_distortion = bool(
                cal.get("intrinsic_matrix") and cal.get("distortion_coefficients")
            )
            point_pairs = cal.get("point_pairs") or []
            has_enough_pairs = sum(
                1
                for pair in point_pairs
                if isinstance(pair, dict) and pair.get("camera_point") and pair.get("plan_point")
            ) >= 4
            single_ready = has_homography and has_floor_plan
            multicam_ready = single_ready and has_coverage
            camera_checks.append({
                "camera_id": camera_id,
                "single_ready": single_ready,
                "multicam_ready": multicam_ready,
                "has_homography": has_homography,
                "has_floor_plan": has_floor_plan,
                "has_distortion": has_distortion,
                "has_enough_pairs": has_enough_pairs,
                "has_coverage": has_coverage,
            })

        total_cameras = len(camera_checks)
        single_camera_ready_count = sum(1 for item in camera_checks if item["single_ready"])
        multicam_ready_count = sum(1 for item in camera_checks if item["multicam_ready"])
        site_spatial_ready = total_cameras > 0 and single_camera_ready_count == total_cameras
        return JSONResponse({
            "ok": True,
            "site_spatial_ready": site_spatial_ready,
            "total_cameras": total_cameras,
            "single_camera_ready_count": single_camera_ready_count,
            "multicam_ready_count": multicam_ready_count,
            "checks": camera_checks,
        })

    @app.post("/api/save-site")
    async def save_site_settings() -> JSONResponse:
        return JSONResponse({"ok": True, "detail": "not implemented in bpa-vision"})

    return app
