"""CLI entry point for calibration tool.

Usage:
  python -m tools.calibration --configs-dir configs --sources-file configs/sources.local.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .app import CalibrationRuntime, create_calibration_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BPA Vision Calibration Tool (camera distortion + homography)"
    )
    parser.add_argument(
        "--configs-dir",
        default="configs",
        help="Directory with calibration YAML files and floor_plans/ subdirectory",
    )
    parser.add_argument(
        "--sources-file",
        default="configs/sources.local.json",
        help="JSON file with [{camera_id, stream_uri}] for RTSP capture",
    )
    parser.add_argument(
        "--camera-frame-width", type=int, default=2560,
        help="Target frame width for calibration (default: 2560)",
    )
    parser.add_argument(
        "--camera-frame-height", type=int, default=1440,
        help="Target frame height for calibration (default: 1440)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args(argv)

    configs_dir = Path(args.configs_dir)
    floor_plans_dir = configs_dir / "floor_plans"
    floor_plans_dir.mkdir(parents=True, exist_ok=True)

    # Load camera URLs from sources file
    camera_urls: dict[str, str] = {}
    sources_path = Path(args.sources_file)
    if sources_path.exists():
        with open(sources_path, "r", encoding="utf-8") as f:
            sources = json.load(f)
        for src in sources:
            camera_urls[src["camera_id"]] = src["stream_uri"]

    runtime = CalibrationRuntime(
        calibration_dir=configs_dir,
        floor_plans_dir=floor_plans_dir,
        camera_urls=camera_urls,
        camera_urls_path=sources_path,
        camera_frame_width=args.camera_frame_width,
        camera_frame_height=args.camera_frame_height,
    )

    app = create_calibration_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
