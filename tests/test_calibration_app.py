"""Tests for calibration FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from tools.calibration.app import CalibrationRuntime, create_calibration_app


def _runtime(tmp_path: Path) -> CalibrationRuntime:
    calibration_dir = tmp_path / "configs"
    floor_plans_dir = calibration_dir / "floor_plans"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    floor_plans_dir.mkdir(parents=True, exist_ok=True)
    return CalibrationRuntime(
        calibration_dir=calibration_dir,
        floor_plans_dir=floor_plans_dir,
        camera_urls={"cam-01": "rtsp://example/stream"},
        camera_urls_path=calibration_dir / "sources.local.json",
    )


class TestCalibrationApi:
    def test_compute_homography_returns_overlay_and_mask(self, tmp_path):
        client = TestClient(create_calibration_app(_runtime(tmp_path)))

        response = client.post(
            "/api/compute-homography",
            json={
                "src_points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                "dst_points": [[100.0, 50.0], [140.0, 50.0], [140.0, 90.0], [100.0, 90.0]],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["matrix"]) == 3
        assert len(body["reprojected_points"]) == 4
        assert body["inlier_mask"] == [True, True, True, True]
        assert body["inliers"] == 4
        assert body["reprojection_error"] == 0.0

    def test_save_persists_full_calibration_contract_and_readiness(self, tmp_path):
        runtime = _runtime(tmp_path)
        client = TestClient(create_calibration_app(runtime))

        payload = {
            "camera_id": "cam-01",
            "rtsp_url": "rtsp://override/stream",
            "anchor_point": "bottom_center",
            "floor_plan_filename": "plan.png",
            "frame_width": 2560,
            "frame_height": 1440,
            "matrix": [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
            "matrix_ground": [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
            "matrix_hip": [[1.0, 0.0, 11.0], [0.0, 1.0, 21.0], [0.0, 0.0, 1.0]],
            "matrix_head": [[1.0, 0.0, 12.0], [0.0, 1.0, 22.0], [0.0, 0.0, 1.0]],
            "intrinsic_matrix": [[1000.0, 0.0, 500.0], [0.0, 1000.0, 300.0], [0.0, 0.0, 1.0]],
            "distortion_coefficients": [0.1, 0.01, 0.0, 0.0, 0.0],
            "distortion_correction_mode": "frame_preprocess",
            "coverage_polygon": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]],
            "point_pairs": [
                {"camera_point": [1.0, 1.0], "plan_point": [10.0, 10.0]},
                {"camera_point": [2.0, 2.0], "plan_point": [20.0, 20.0]},
                {"camera_point": [3.0, 3.0], "plan_point": [30.0, 30.0]},
                {"camera_point": [4.0, 4.0], "plan_point": [40.0, 40.0]},
            ],
            "homography_stats": {
                "reprojection_error": 0.5,
                "inlier_reprojection_error": 0.25,
                "inliers": 4,
                "inlier_mask": [True, True, True, True],
            },
        }

        save_response = client.post("/api/save", json=payload)
        assert save_response.status_code == 200
        save_body = save_response.json()
        assert save_body["saved_planes"] == {
            "base": True,
            "ground": True,
            "hip": True,
            "head": True,
        }

        yaml_path = runtime.calibration_dir / "calibration_cam-01.yaml"
        saved = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert saved["camera_id"] == "cam-01"
        assert saved["anchor_point"] == "bottom_center"
        assert saved["homography_ground"] == payload["matrix_ground"]
        assert saved["coverage_polygon"] == payload["coverage_polygon"]
        assert saved["point_pairs"] == payload["point_pairs"]
        assert saved["homography_stats"]["inliers"] == 4

        cameras_response = client.get("/api/cameras")
        assert cameras_response.status_code == 200
        camera_meta = cameras_response.json()["cameras"][0]
        assert camera_meta["anchor_point"] == "bottom_center"
        assert camera_meta["homography_ground"] == payload["matrix_ground"]
        assert camera_meta["point_pairs"] == payload["point_pairs"]
        assert camera_meta["has_homography"] is True

        readiness_response = client.get("/api/readiness")
        assert readiness_response.status_code == 200
        readiness = readiness_response.json()
        assert readiness["total_cameras"] == 1
        assert readiness["single_camera_ready_count"] == 1
        assert readiness["multicam_ready_count"] == 1
        assert readiness["site_spatial_ready"] is True
