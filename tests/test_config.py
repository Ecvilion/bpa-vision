"""Tests for YAML config loader and schema validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from bpa_vision.config.loader import load_config
from bpa_vision.config.schema import SiteConfig


MINIMAL_CONFIG = {
    "site_id": "test-site",
    "site_name": "Test Site",
}

FULL_CONFIG = {
    "site_id": "test-site",
    "site_name": "Test Site",
    "cameras": [
        {
            "camera_id": "cam1",
            "stream_uri": "rtsp://localhost/stream1",
            "tracking": {"profile": "strong", "t_lost_seconds": 10.0},
            "zones": [
                {
                    "zone_id": "z1",
                    "zone_type": "polygon",
                    "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                }
            ],
            "analytics": {
                "rules": [
                    {
                        "rule_id": "r1",
                        "rule_type": "dwell",
                        "zone_id": "z1",
                        "params": {"max_dwell_seconds": 60},
                    }
                ]
            },
        }
    ],
    "identity": {
        "confirm_threshold": 0.8,
    },
    "retention": {
        "tracks_days": 7,
    },
}


def _write_yaml(data: dict, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return path


class TestLoadConfig:
    def test_minimal_config(self, tmp_path):
        cfg_path = _write_yaml(MINIMAL_CONFIG, tmp_path / "config.yaml")
        cfg = load_config(cfg_path)
        assert cfg.site_id == "test-site"
        assert cfg.cameras == []
        assert cfg.identity.confirm_threshold == 0.75  # default

    def test_full_config(self, tmp_path):
        cfg_path = _write_yaml(FULL_CONFIG, tmp_path / "config.yaml")
        cfg = load_config(cfg_path)
        assert len(cfg.cameras) == 1
        cam = cfg.cameras[0]
        assert cam.camera_id == "cam1"
        assert cam.tracking.profile.value == "strong"
        assert len(cam.zones) == 1
        assert len(cam.analytics.rules) == 1
        assert cfg.identity.confirm_threshold == 0.8
        assert cfg.retention.tracks_days == 7

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_invalid_yaml_content(self, tmp_path):
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("- just a list\n- not a mapping\n")
        with pytest.raises(ValueError, match="Expected YAML mapping"):
            load_config(bad_path)

    def test_invalid_schema(self, tmp_path):
        invalid = {"site_id": "s1"}  # missing site_name
        cfg_path = _write_yaml(invalid, tmp_path / "config.yaml")
        with pytest.raises(Exception):
            load_config(cfg_path)

    def test_example_config_loads(self):
        """The shipped example config must always be valid."""
        cfg = load_config("configs/example.yaml")
        assert cfg.site_id == "site-001"
        assert len(cfg.cameras) == 2
