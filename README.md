# BPA Vision

Real-time multi-camera pose detection pipeline built on NVIDIA DeepStream SDK.

## Architecture

RTSP cameras → DeepStream (nvdewarper [optional] → nvstreammux → nvinfer/YOLO26-pose → nvdsosd → tiler) → MP4 recording + TCP live stream + JSONL observations.

Working resolution: 2560x1440. All output coordinates are in normalized [0,1] space.

## Prerequisites

- NVIDIA GPU with driver >= 535
- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- YOLO26m-pose ONNX model in `models/yolo26m-pose.onnx`

## Quick Start

1. **Configure cameras** — copy the template and add your credentials:

   ```bash
   cp configs/sources.json configs/sources.local.json
   # Edit configs/sources.local.json with real RTSP URIs
   ```

2. **Create a docker-compose override** to use your local config:

   ```yaml
   # docker-compose.override.yml
   services:
     deepstream:
       command: >
         python3 /app/deepstream/pipeline.py
           --sources-file /app/configs/sources.local.json
           --pose-config /app/deepstream/config_infer_yolo26m_pose.yml
           --output-dir /app/output
           --conf-threshold 0.25
           --stream-port 5555
           --calibration-dir /app/configs
   ```

3. **Run:**

   ```bash
   docker compose up --build
   ```

4. **View live stream** (from host):

   ```python
   import cv2
   cap = cv2.VideoCapture("tcp://localhost:5555")
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       cv2.imshow("BPA Vision", frame)
       if cv2.waitKey(1) & 0xFF == ord("q"):
           break
   ```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--sources-file` | — | JSON file with `[{camera_id, stream_uri}]` |
| `--sources` | — | Inline RTSP URIs (alternative to file) |
| `--pose-config` | `/app/deepstream/config_infer_yolo26m_pose.yml` | nvinfer config path |
| `--output-dir` | `/app/output` | Directory for MP4 and JSONL files |
| `--conf-threshold` | `0.25` | Detection confidence threshold |
| `--stream-port` | `0` | TCP stream port (`0` = disabled) |
| `--segment-duration` | `60` | MP4 segment length in seconds |
| `--record-duration` | `0` | Stop pipeline after N seconds (`0` = unlimited) |
| `--calibration-dir` | — | Dir with `calibration_<cam_id>.yaml` for GPU undistortion |

## Undistortion

If `--calibration-dir` is set and a file `calibration_<camera_id>.yaml` exists with `intrinsic_matrix` and `distortion_coefficients`, the pipeline inserts an `nvdewarper` element per source for GPU-accelerated lens distortion correction. If no calibration is found, the source passes through without overhead.

## Calibration Tool

Debug utility for camera calibration (distortion + homography). Not required at runtime.

```bash
pip install fastapi uvicorn opencv-python pyyaml
python -m tools.calibration --configs-dir configs --sources-file configs/sources.local.json
# Open http://localhost:8098
```

## Outputs

- `output/output_00.mp4`, `output_01.mp4`, ... — tiled video segments
- `output/<camera_id>_observations.jsonl` — per-camera pose detections in normalized [0,1] coordinates

## Tests

```bash
pip install pytest
pytest tests/
```
