"""Tests for calibration FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
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
    def test_line_based_distortion_endpoint_returns_model(self, tmp_path):
        client = TestClient(create_calibration_app(_runtime(tmp_path)))

        response = client.post(
            "/api/distortion/line-based",
            json={
                "lines": [
                    [[200.0, 120.0], [360.0, 140.0], [540.0, 160.0], [760.0, 185.0]],
                    [[260.0, 520.0], [420.0, 420.0], [580.0, 320.0], [760.0, 210.0]],
                ],
                "image_width": 1280,
                "image_height": 720,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["model"]["model"] == "line_based_radial_v1"
        assert len(body["line_errors"]) == 2
        assert body["line_count"] == 2
        assert body["total_points"] == 8

    def test_undistort_points_endpoint_preserves_nulls_and_marks_space(self, tmp_path):
        client = TestClient(create_calibration_app(_runtime(tmp_path)))

        response = client.post(
            "/api/points/undistort",
            json={
                "points": [[100.0, 120.0], None, [320.0, 240.0]],
                "intrinsic_matrix": [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "image_width": 640,
                "image_height": 480,
                "alpha": 0.0,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["camera_points_space"] == "undistorted"
        assert body["points"][1] is None
        assert body["points"][0][0] == pytest.approx(100.0, abs=1e-4)
        assert body["points"][0][1] == pytest.approx(120.0, abs=1e-4)
        assert body["points"][2][0] == pytest.approx(320.0, abs=1e-4)
        assert body["points"][2][1] == pytest.approx(240.0, abs=1e-4)
        assert len(body["new_intrinsic_matrix"]) == 3
        assert len(body["roi"]) == 4
        assert body["roi"][2] > 0
        assert body["roi"][3] > 0

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

    def test_compute_homography_accepts_runtime_ransac_parameters(self, tmp_path):
        client = TestClient(create_calibration_app(_runtime(tmp_path)))

        response = client.post(
            "/api/compute-homography",
            json={
                "src_points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                "dst_points": [[100.0, 50.0], [140.0, 50.0], [140.0, 90.0], [100.0, 90.0]],
                "homography_method": "all_points",
                "ransac_reproj_threshold": 8.0,
                "confidence": 0.995,
                "max_iterations": 5000,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["homography_method"] == "all_points"
        assert body["estimator"] == "ALL_POINTS"
        assert body["ransac_reproj_threshold"] == 8.0
        assert body["confidence"] == 0.995
        assert body["max_iterations"] == 5000

    def test_save_persists_full_calibration_contract_and_readiness(self, tmp_path):
        runtime = _runtime(tmp_path)
        client = TestClient(create_calibration_app(runtime))

        payload = {
            "camera_id": "cam-01",
            "rtsp_url": "rtsp://override/stream",
            "anchor_point": "bottom_center",
            "camera_points_space": "undistorted",
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
            "line_constraints": [
                [[100.0, 100.0], [200.0, 110.0], [300.0, 120.0]],
                [[120.0, 240.0], [240.0, 220.0], [360.0, 200.0]],
            ],
            "line_based_distortion": {
                "model": "line_based_radial_v1",
                "k1": 0.2,
                "center_norm": [0.5, 0.5],
                "source_image_width": 2560,
                "source_image_height": 1440,
            },
            "line_based_stats": {
                "mean_line_error": 0.75,
                "max_line_error": 1.25,
                "line_errors": [0.5, 1.0],
                "line_count": 2,
                "total_points": 6,
            },
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
        assert saved["camera_points_space"] == "undistorted"
        assert saved["homography_ground"] == payload["matrix_ground"]
        assert saved["coverage_polygon"] == payload["coverage_polygon"]
        assert saved["point_pairs"] == payload["point_pairs"]
        assert saved["line_constraints"] == payload["line_constraints"]
        assert saved["line_based_distortion"]["k1"] == 0.2
        assert saved["line_based_stats"]["line_count"] == 2
        assert saved["homography_stats"]["inliers"] == 4

        cameras_response = client.get("/api/cameras")
        assert cameras_response.status_code == 200
        camera_meta = cameras_response.json()["cameras"][0]
        assert camera_meta["anchor_point"] == "bottom_center"
        assert camera_meta["camera_points_space"] == "undistorted"
        assert camera_meta["homography_ground"] == payload["matrix_ground"]
        assert camera_meta["point_pairs"] == payload["point_pairs"]
        assert camera_meta["line_constraints"] == payload["line_constraints"]
        assert camera_meta["line_based_distortion"]["k1"] == 0.2
        assert camera_meta["has_homography"] is True

        readiness_response = client.get("/api/readiness")
        assert readiness_response.status_code == 200
        readiness = readiness_response.json()
        assert readiness["total_cameras"] == 1
        assert readiness["single_camera_ready_count"] == 1
        assert readiness["multicam_ready_count"] == 1
        assert readiness["site_spatial_ready"] is True
