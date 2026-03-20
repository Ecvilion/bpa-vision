"""Parse YOLO26 end-to-end model output tensors from DeepStream nvinfer.

Detection model output:  [1, 300, 6]  -> [x1, y1, x2, y2, confidence, class_id]
Pose model output:       [1, 300, 57] -> [x1, y1, x2, y2, conf, class_id, kp0_x, kp0_y, kp0_conf, ...]

Both models have built-in NMS — output is top-300 detections sorted by confidence,
padded with zeros for unused slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyds


# COCO person class id
PERSON_CLASS_ID = 0

# Confidence threshold for filtering padded/low-confidence detections
DEFAULT_CONF_THRESHOLD = 0.25

# 17 COCO keypoints
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17


@dataclass
class Detection:
    """Parsed person detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


@dataclass
class PoseDetection(Detection):
    """Parsed person detection with pose keypoints."""
    keypoints: list[tuple[float, float, float]] = field(default_factory=list)


def _get_tensor_output(tensor_meta) -> np.ndarray | None:
    """Extract numpy array from NvDsInferTensorMeta output layer."""
    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    if layer is None:
        return None

    dims = layer.inferDims
    # Build shape from dims
    num_dims = dims.numDims
    shape = [dims.d[i] for i in range(num_dims)]

    ptr = pyds.get_ptr(layer.buffer)
    if ptr is None:
        return None

    # Total elements
    total = 1
    for s in shape:
        total *= s

    arr = np.ctypeslib.as_array(ptr, shape=(total,))
    return arr.reshape(shape).copy()


def parse_detection_tensor(
    tensor_meta,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    input_width: int = 640,
    input_height: int = 640,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> list[Detection]:
    """Parse detection model output tensor [300, 6] into Detection list.

    Coordinates are returned in pixel space of the original frame.
    """
    arr = _get_tensor_output(tensor_meta)
    if arr is None:
        return []

    # Squeeze batch dim if present -> [300, 6]
    if arr.ndim == 3:
        arr = arr[0]

    detections = []
    scale_x = frame_width / input_width
    scale_y = frame_height / input_height

    for i in range(arr.shape[0]):
        row = arr[i]
        conf = float(row[4])
        if conf < conf_threshold:
            continue

        class_id = int(row[5])
        if class_id != PERSON_CLASS_ID:
            continue

        x1 = float(row[0]) * scale_x
        y1 = float(row[1]) * scale_y
        x2 = float(row[2]) * scale_x
        y2 = float(row[3]) * scale_y

        detections.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=conf, class_id=class_id,
        ))

    return detections


def parse_pose_tensor(
    tensor_meta,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    kp_conf_threshold: float = 0.3,
    input_width: int = 640,
    input_height: int = 640,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> list[PoseDetection]:
    """Parse pose model output tensor [300, 57] into PoseDetection list.

    Each row: [x1, y1, x2, y2, conf, class_id, kp0_x, kp0_y, kp0_conf, ...]
    Coordinates are returned in pixel space of the original frame.
    """
    arr = _get_tensor_output(tensor_meta)
    if arr is None:
        return []

    if arr.ndim == 3:
        arr = arr[0]

    detections = []
    scale_x = frame_width / input_width
    scale_y = frame_height / input_height

    for i in range(arr.shape[0]):
        row = arr[i]
        conf = float(row[4])
        if conf < conf_threshold:
            continue

        class_id = int(row[5])

        x1 = float(row[0]) * scale_x
        y1 = float(row[1]) * scale_y
        x2 = float(row[2]) * scale_x
        y2 = float(row[3]) * scale_y

        # Parse 17 keypoints starting at index 6
        keypoints = []
        for k in range(NUM_KEYPOINTS):
            base = 6 + k * 3
            kp_x = float(row[base]) * scale_x
            kp_y = float(row[base + 1]) * scale_y
            kp_c = float(row[base + 2])
            keypoints.append((kp_x, kp_y, kp_c))

        detections.append(PoseDetection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=conf, class_id=class_id,
            keypoints=keypoints,
        ))

    return detections


def detections_to_normalized(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    """Convert pixel-space detections to FrameNormalizedSpace [0,1]."""
    results = []
    for det in detections:
        entry = {
            "bbox": {
                "x_min": det.x1 / frame_width,
                "y_min": det.y1 / frame_height,
                "x_max": det.x2 / frame_width,
                "y_max": det.y2 / frame_height,
            },
            "confidence": det.confidence,
            "class_id": det.class_id,
        }
        if isinstance(det, PoseDetection) and det.keypoints:
            entry["keypoints"] = [
                {
                    "x": kp[0] / frame_width,
                    "y": kp[1] / frame_height,
                    "confidence": kp[2],
                }
                for kp in det.keypoints
            ]
        results.append(entry)
    return results
