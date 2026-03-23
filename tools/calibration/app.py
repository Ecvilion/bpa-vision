"""Calibration web tool — FastAPI app for camera distortion & homography.

Debug utility for camera setup. NOT a production dependency.
Run standalone: python -m tools.calibration --configs-dir configs

Ported from vision_box.tools.calibration.app — stripped of vision_box deps.
"""

from __future__ import annotations

import json
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from .distortion import (
    calibrate_camera_from_chessboard_images,
    decode_image_bytes,
    undistort_image,
)
from .homography import _compute_homography_impl, reproject_points
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


class SaveCalibrationRequest(BaseModel):
    camera_id: str
    matrix: list[list[float]] | None = None
    floor_plan_filename: str = ""
    frame_width: int | None = None
    frame_height: int | None = None
    intrinsic_matrix: list[list[float]] | None = None
    distortion_coefficients: list[float] | None = None
    distortion_correction_mode: str | None = None


class CaptureFrameRequest(BaseModel):
    camera_id: str | None = None
    rtsp_url: str | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    intrinsic_matrix: list[list[float]] | None = None
    distortion_coefficients: list[float] | None = None
    apply_undistort: bool = False
    undistort_alpha: float = 0.0


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
        })

    @app.get("/api/cameras")
    async def cameras() -> JSONResponse:
        out: list[dict[str, Any]] = []
        for camera_id, rtsp_url in runtime.camera_urls.items():
            cal = runtime.load_camera_calibration(camera_id)
            matrix = cal.get("homography_matrix")
            out.append({
                "camera_id": camera_id,
                "has_homography": bool(not _is_identity_homography(matrix)),
                "rtsp_url": rtsp_url,
                "homography_matrix": matrix,
                "floor_plan_image": cal.get("floor_plan_image"),
                "frame_width": cal.get("frame_width"),
                "frame_height": cal.get("frame_height"),
                "intrinsic_matrix": cal.get("intrinsic_matrix"),
                "distortion_coefficients": cal.get("distortion_coefficients"),
                "distortion_correction_mode": cal.get("distortion_correction_mode", "none"),
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

    @app.post("/api/capture-frame")
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

        if req.apply_undistort and req.intrinsic_matrix and req.distortion_coefficients:
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

    @app.post("/api/compute-homography")
    async def compute_homography_endpoint(req: ComputeHomographyRequest) -> JSONResponse:
        try:
            matrix, error, inliers, estimator = _compute_homography_impl(
                req.src_points, req.dst_points
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse({
            "matrix": matrix,
            "reprojection_error": error,
            "inliers": inliers,
            "estimator": estimator,
        })

    @app.post("/api/save-calibration")
    async def save_calibration(req: SaveCalibrationRequest) -> JSONResponse:
        cam_id = req.camera_id.strip()
        if not cam_id:
            raise HTTPException(status_code=400, detail="camera_id required")

        cal = runtime.load_camera_calibration(cam_id)
        if req.matrix:
            validate_homography_matrix(req.matrix, cam_id)
            cal["homography_matrix"] = req.matrix
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

        path = runtime.save_camera_calibration(cam_id, cal)
        return JSONResponse({
            "ok": True,
            "path": str(path),
            "saved_at": _utcnow().isoformat(),
        })

    @app.post("/api/calibrate-distortion")
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

    return app
