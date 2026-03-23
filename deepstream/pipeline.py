"""DeepStream Python pipeline for BPA Vision.

Pipeline (per source):
  rtspsrc → depay → queue → parse → nvv4l2decoder → queue →
  [nvdewarper → queue] (optional, if calibration present) →
  nvstreammux(batch=1) → queue → nvinfer (pose) → queue →
  nvvideoconvert → queue → nvdsosd → queue →
  nvmultistreamtiler → queue → tee
    ├→ queue → nvvideoconvert → capsfilter → nvv4l2h264enc →
    │  h264parse → splitmuxsink
    └→ queue → nvvideoconvert → capsfilter → nvv4l2h264enc →
       h264parse → matroskamux → tcpserversink

Python probe on nvinfer src pad:
  1. Parse output tensors → PoseDetection (pixel-space)
  2. Attach bbox/keypoints as OSD display meta
  3. Convert to normalized [0,1] and write JSONL records

Usage:
  python3 deepstream/pipeline.py --sources-file configs/sources.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse
from uuid import uuid4

import yaml
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds

from yolo26_parser import (
    NUM_KEYPOINTS,
    PoseDetection,
    parse_pose_tensor,
    detections_to_normalized,
)
from dewarper import render_dewarper_config

logger = logging.getLogger("bpa_vision.pipeline")

# ── Constants ──────────────────────────────────────────────────────────────
MUXER_WIDTH = 2560
MUXER_HEIGHT = 1440
MUXER_BATCH_TIMEOUT = 40000  # μs
GPU_ID = 0

# COCO skeleton for OSD drawing (pairs of keypoint indices)
MAX_OSD_ELEMENTS = 16  # pyds.MAX_ELEMENTS_IN_DISPLAY_META

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


# ── Helpers ────────────────────────────────────────────────────────────────

def make_element(factory: str, name: str) -> Gst.Element:
    """Create a GStreamer element or raise."""
    elem = Gst.ElementFactory.make(factory, name)
    if elem is None:
        raise RuntimeError(f"Failed to create element: {factory} ({name})")
    return elem


def make_queue(name: str) -> Gst.Element:
    """Create a queue element with reasonable defaults."""
    q = make_element("queue", name)
    q.set_property("max-size-buffers", 4)
    q.set_property("leaky", 2)  # leak downstream
    return q


def _load_calibration(calibration_dir: str, camera_id: str) -> dict | None:
    """Load calibration YAML for a camera, or return None if not found."""
    cal_dir = Path(calibration_dir)
    cal_path = cal_dir / f"calibration_{camera_id}.yaml"
    if not cal_path.exists():
        return None
    try:
        with open(cal_path, "r", encoding="utf-8") as f:
            cal = yaml.safe_load(f)
        if (
            isinstance(cal, dict)
            and cal.get("intrinsic_matrix")
            and cal.get("distortion_coefficients")
        ):
            return cal
    except Exception as e:
        logger.warning("Failed to load calibration %s: %s", cal_path, e)
    return None


class PipelineManager:
    """Builds and runs the DeepStream pipeline for multiple RTSP sources."""

    def __init__(
        self,
        sources: list[dict],
        pose_config: str,
        output_dir: str = "/app/output",
        conf_threshold: float = 0.25,
        stream_port: int = 0,
        segment_duration: int = 60,
        record_duration: int = 0,
        calibration_dir: str = "",
    ):
        self.sources = sources
        self.pose_config = pose_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = conf_threshold
        self.stream_port = stream_port  # TCP stream port, 0 = disabled
        self.segment_duration = segment_duration  # MP4 segment length in seconds
        self.record_duration = record_duration  # total recording time, 0 = unlimited
        self.calibration_dir = calibration_dir  # dir with calibration_<cam_id>.yaml

        self.pipeline: Gst.Pipeline | None = None
        self.loop: GLib.MainLoop | None = None
        self.jsonl_files: dict[int, object] = {}
        self.frame_counters: dict[int, int] = {}

    # ── Build pipeline ─────────────────────────────────────────────────

    def build(self) -> Gst.Pipeline:
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("bpa-vision-pipeline")

        # ── Streammux ──
        streammux = make_element("nvstreammux", "muxer")
        streammux.set_property("batch-size", 1)
        streammux.set_property("width", MUXER_WIDTH)
        streammux.set_property("height", MUXER_HEIGHT)
        streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT)
        streammux.set_property("gpu-id", GPU_ID)
        streammux.set_property("nvbuf-memory-type", 0)  # default
        self.pipeline.add(streammux)

        # ── Source branches ──
        for idx, src in enumerate(self.sources):
            self._add_source_branch(idx, src, streammux)

        # ── Queue after muxer ──
        q_mux = make_queue("q_after_mux")
        self.pipeline.add(q_mux)
        streammux.link(q_mux)

        # ── nvinfer (pose) ──
        nvinfer = make_element("nvinfer", "pose-infer")
        nvinfer.set_property("config-file-path", self.pose_config)
        nvinfer.set_property("gpu-id", GPU_ID)
        self.pipeline.add(nvinfer)
        q_inf = make_queue("q_after_infer")
        self.pipeline.add(q_inf)
        q_mux.link(nvinfer)
        nvinfer.link(q_inf)

        # ── nvvideoconvert (for OSD) ──
        nvvidconv = make_element("nvvideoconvert", "nvvidconv")
        nvvidconv.set_property("gpu-id", GPU_ID)
        self.pipeline.add(nvvidconv)
        q_conv = make_queue("q_after_conv")
        self.pipeline.add(q_conv)
        q_inf.link(nvvidconv)
        nvvidconv.link(q_conv)

        # ── nvdsosd ──
        osd = make_element("nvdsosd", "osd")
        osd.set_property("process-mode", 1)  # GPU
        osd.set_property("gpu-id", GPU_ID)
        self.pipeline.add(osd)
        q_osd = make_queue("q_after_osd")
        self.pipeline.add(q_osd)
        q_conv.link(osd)
        osd.link(q_osd)

        # ── nvmultistreamtiler (all cameras side by side) ──
        num_src = len(self.sources)
        tiler = make_element("nvmultistreamtiler", "tiler")
        tiler.set_property("rows", 1)
        tiler.set_property("columns", num_src)
        tiler.set_property("width", MUXER_WIDTH)
        tiler.set_property("height", MUXER_HEIGHT)
        tiler.set_property("gpu-id", GPU_ID)
        self.pipeline.add(tiler)
        q_tiler = make_queue("q_after_tiler")
        self.pipeline.add(q_tiler)
        q_osd.link(tiler)
        tiler.link(q_tiler)

        # ── tee ──
        tee = make_element("tee", "tee")
        self.pipeline.add(tee)
        q_tiler.link(tee)

        # ── Branch 1: encode → splitmuxsink (MP4 recording) ──
        q_enc = make_queue("q_encode")
        nvvidconv_enc = make_element("nvvideoconvert", "nvvidconv_enc")
        nvvidconv_enc.set_property("gpu-id", GPU_ID)
        q_caps_enc = make_queue("q_caps_enc")
        capsfilter = make_element("capsfilter", "caps_enc")
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        )
        q_encoder = make_queue("q_encoder")
        encoder = make_element("nvv4l2h264enc", "encoder")
        encoder.set_property("bitrate", 8000000)
        q_h264parse = make_queue("q_h264parse")
        h264parse = make_element("h264parse", "h264parse")
        splitmux = make_element("splitmuxsink", "splitmux")
        splitmux.set_property(
            "location", str(self.output_dir / "output_%02d.mp4"),
        )
        splitmux.set_property("max-size-time", self.segment_duration * Gst.SECOND)

        for el in [
            q_enc, nvvidconv_enc, q_caps_enc, capsfilter,
            q_encoder, encoder, q_h264parse, h264parse, splitmux,
        ]:
            self.pipeline.add(el)

        tee.link(q_enc)
        q_enc.link(nvvidconv_enc)
        nvvidconv_enc.link(q_caps_enc)
        q_caps_enc.link(capsfilter)
        capsfilter.link(q_encoder)
        q_encoder.link(encoder)
        encoder.link(q_h264parse)
        q_h264parse.link(h264parse)
        h264parse.link(splitmux)

        # ── Branch 2: TCP stream or fakesink ──
        if self.stream_port > 0:
            self._add_stream_branch(tee)
        else:
            q_fake = make_queue("q_fake")
            fakesink = make_element("fakesink", "fakesink")
            fakesink.set_property("sync", 0)
            self.pipeline.add(q_fake)
            self.pipeline.add(fakesink)
            tee.link(q_fake)
            q_fake.link(fakesink)

        # ── Probe on nvinfer src pad to parse tensors + draw OSD ──
        infer_src_pad = nvinfer.get_static_pad("src")
        if infer_src_pad:
            infer_src_pad.add_probe(
                Gst.PadProbeType.BUFFER, self._infer_src_probe, None
            )

        # ── Open JSONL files ──
        for idx, src in enumerate(self.sources):
            cam_id = src.get("camera_id", f"cam_{idx}")
            jsonl_path = self.output_dir / f"{cam_id}_observations.jsonl"
            self.jsonl_files[idx] = open(jsonl_path, "w", encoding="utf-8")
            self.frame_counters[idx] = 0

        return self.pipeline

    def _add_source_branch(
        self, idx: int, src: dict, streammux: Gst.Element
    ) -> None:
        """Add one RTSP source → decode → [dewarper] → muxer sink pad."""
        uri = src["stream_uri"]
        cam_id = src.get("camera_id", f"cam_{idx}")

        # GStreamer rtspsrc does NOT url-decode passwords in inline URIs,
        # so special characters (like !) must be percent-encoded.
        parsed = urlparse(uri)
        if parsed.username and parsed.password:
            encoded_user = quote(parsed.username, safe="")
            encoded_pass = quote(parsed.password, safe="")
            netloc = f"{encoded_user}:{encoded_pass}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            uri = urlunparse(parsed._replace(netloc=netloc))

        rtspsrc = make_element("rtspsrc", f"rtspsrc_{idx}")
        rtspsrc.set_property("location", uri)
        rtspsrc.set_property("latency", 200)
        rtspsrc.set_property("drop-on-latency", True)
        # Force TCP to avoid UDP firewall issues
        rtspsrc.set_property("protocols", 4)  # GST_RTSP_LOWER_TRANS_TCP

        decoder = make_element("nvv4l2decoder", f"decoder_{idx}")
        decoder.set_property("gpu-id", GPU_ID)
        q_dec = make_queue(f"q_decode_{idx}")

        for el in [rtspsrc, decoder, q_dec]:
            self.pipeline.add(el)

        # rtspsrc has dynamic pads — connect on pad-added with auto depay
        rtspsrc.connect(
            "pad-added", self._on_rtspsrc_pad_added_auto, idx, decoder
        )
        decoder.link(q_dec)

        # ── Optional nvdewarper (GPU undistortion) ──
        last_element = q_dec
        if self.calibration_dir:
            cal = _load_calibration(self.calibration_dir, cam_id)
            if cal:
                try:
                    dw_cfg = render_dewarper_config(
                        camera_id=cam_id,
                        intrinsic_matrix=cal["intrinsic_matrix"],
                        distortion_coefficients=cal["distortion_coefficients"],
                        width=cal.get("frame_width", MUXER_WIDTH),
                        height=cal.get("frame_height", MUXER_HEIGHT),
                        output_dir=str(self.output_dir / "dewarper_configs"),
                    )
                    dewarper = make_element("nvdewarper", f"dewarper_{idx}")
                    dewarper.set_property("config-file", dw_cfg.config_file)
                    dewarper.set_property("gpu-id", GPU_ID)
                    q_dw = make_queue(f"q_dewarper_{idx}")
                    self.pipeline.add(dewarper)
                    self.pipeline.add(q_dw)
                    q_dec.link(dewarper)
                    dewarper.link(q_dw)
                    last_element = q_dw
                    logger.info(
                        "Source %d (%s): nvdewarper enabled (config: %s)",
                        idx, cam_id, dw_cfg.config_file,
                    )
                except Exception as e:
                    logger.warning(
                        "Source %d (%s): nvdewarper setup failed, skipping: %s",
                        idx, cam_id, e,
                    )

        # Request a sink pad on streammux
        pad_name = f"sink_{idx}"
        mux_sink = streammux.request_pad_simple(pad_name)
        last_src_pad = last_element.get_static_pad("src")
        last_src_pad.link(mux_sink)

    def _on_rtspsrc_pad_added_auto(
        self, rtspsrc: Gst.Element, pad: Gst.Pad, idx: int, decoder: Gst.Element
    ) -> None:
        """Auto-detect codec from rtspsrc pad and insert correct depay+parse."""
        caps = pad.get_current_caps()
        if caps is None:
            return
        struct = caps.get_structure(0)
        name = struct.get_name()
        if not name.startswith("application/x-rtp"):
            return

        # Check media type — ignore audio streams
        media = struct.get_string("media") or ""
        if media != "video":
            return

        encoding = struct.get_string("encoding-name") or ""
        encoding = encoding.upper()
        logger.info("Source %d: detected codec %s", idx, encoding)

        if encoding == "H264":
            depay = make_element("rtph264depay", f"depay_{idx}")
            parse = make_element("h264parse", f"h264parse_src_{idx}")
        elif encoding in ("H265", "HEVC"):
            depay = make_element("rtph265depay", f"depay_{idx}")
            parse = make_element("h265parse", f"h265parse_src_{idx}")
        else:
            logger.warning("Source %d: unsupported video codec %s", idx, encoding)
            return

        q_depay = make_queue(f"q_depay_{idx}")
        self.pipeline.add(depay)
        self.pipeline.add(q_depay)
        self.pipeline.add(parse)

        pad.link(depay.get_static_pad("sink"))
        depay.link(q_depay)
        q_depay.link(parse)
        parse.link(decoder)

        depay.sync_state_with_parent()
        q_depay.sync_state_with_parent()
        parse.sync_state_with_parent()

    # ── Probe callback ─────────────────────────────────────────────────

    def _infer_src_probe(
        self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data
    ) -> Gst.PadProbeReturn:
        """Parse nvinfer output tensors, write JSONL, add OSD metadata."""
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id = frame_meta.source_id
            self.frame_counters.setdefault(source_id, 0)
            self.frame_counters[source_id] += 1
            frame_idx = self.frame_counters[source_id]

            # Parse output tensors
            detections = self._extract_pose_detections(frame_meta)

            # Convert to normalized [0,1] space and write JSONL
            if detections:
                normalized = detections_to_normalized(
                    detections, MUXER_WIDTH, MUXER_HEIGHT
                )
                now = datetime.now(timezone.utc).isoformat()
                cam_id = self.sources[source_id].get(
                    "camera_id", f"cam_{source_id}"
                ) if source_id < len(self.sources) else f"cam_{source_id}"

                for det_norm in normalized:
                    record = {
                        "observation_id": str(uuid4()),
                        "camera_id": cam_id,
                        "frame_idx": frame_idx,
                        "timestamp": now,
                        **det_norm,
                    }
                    if source_id in self.jsonl_files:
                        self.jsonl_files[source_id].write(
                            json.dumps(record) + "\n"
                        )

                # Add OSD display meta (bboxes + keypoints)
                self._add_osd_meta(frame_meta, detections)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        # Flush JSONL periodically
        for f in self.jsonl_files.values():
            f.flush()

        return Gst.PadProbeReturn.OK

    def _extract_pose_detections(
        self, frame_meta
    ) -> list[PoseDetection]:
        """Extract pose detections from tensor output meta."""
        detections = []

        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if (
                user_meta.base_meta.meta_type
                == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
            ):
                tensor_meta = pyds.NvDsInferTensorMeta.cast(
                    user_meta.user_meta_data
                )

                # Log tensor info once for debugging
                if not hasattr(self, "_tensor_logged"):
                    self._tensor_logged = True
                    num_layers = tensor_meta.num_output_layers
                    logger.info("Tensor meta: %d output layers", num_layers)
                    for li in range(num_layers):
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, li)
                        if layer:
                            dims = layer.inferDims
                            shape = [dims.d[i] for i in range(dims.numDims)]
                            logger.info(
                                "  Layer %d: name=%s, dims=%s, dataType=%s",
                                li, layer.layerName, shape, layer.dataType,
                            )

                try:
                    detections = parse_pose_tensor(
                        tensor_meta,
                        conf_threshold=self.conf_threshold,
                        input_width=640,
                        input_height=640,
                        frame_width=MUXER_WIDTH,
                        frame_height=MUXER_HEIGHT,
                    )
                except Exception as e:
                    if not hasattr(self, "_parse_error_logged"):
                        self._parse_error_logged = True
                        logger.error("Tensor parse error: %s", e)
                break

            try:
                l_user = l_user.next
            except StopIteration:
                break

        return detections

    def _add_osd_meta(
        self, frame_meta, detections: list[PoseDetection]
    ) -> None:
        """Attach bounding boxes and pose skeleton lines to OSD display meta."""
        display_meta = pyds.nvds_acquire_display_meta_from_pool(
            frame_meta.base_meta.batch_meta
        )

        rect_idx = 0
        line_idx = 0

        for det in detections:
            # Draw bounding box
            if rect_idx < MAX_OSD_ELEMENTS:
                rect = pyds.NvOSD_RectParams.cast(
                    display_meta.rect_params[rect_idx]
                )
                rect.left = max(0, int(det.x1))
                rect.top = max(0, int(det.y1))
                rect.width = max(1, int(det.x2 - det.x1))
                rect.height = max(1, int(det.y2 - det.y1))
                rect.border_width = 2
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)  # green
                rect.has_bg_color = 0
                rect_idx += 1

            # Draw skeleton lines
            if isinstance(det, PoseDetection) and det.keypoints:
                for i, j in SKELETON_PAIRS:
                    kp_i = det.keypoints[i]
                    kp_j = det.keypoints[j]
                    # Only draw if both keypoints are confident
                    if kp_i[2] < 0.3 or kp_j[2] < 0.3:
                        continue
                    if line_idx >= MAX_OSD_ELEMENTS:
                        # Need a new display meta
                        display_meta.num_rects = rect_idx
                        display_meta.num_lines = line_idx
                        pyds.nvds_add_display_meta_to_frame(
                            frame_meta, display_meta
                        )
                        display_meta = pyds.nvds_acquire_display_meta_from_pool(
                            frame_meta.base_meta.batch_meta
                        )
                        rect_idx = 0
                        line_idx = 0

                    line = pyds.NvOSD_LineParams.cast(
                        display_meta.line_params[line_idx]
                    )
                    line.x1 = max(0, int(kp_i[0]))
                    line.y1 = max(0, int(kp_i[1]))
                    line.x2 = max(0, int(kp_j[0]))
                    line.y2 = max(0, int(kp_j[1]))
                    line.line_width = 2
                    line.line_color.set(0.0, 1.0, 1.0, 1.0)  # cyan
                    line_idx += 1

        display_meta.num_rects = rect_idx
        display_meta.num_lines = line_idx
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    # ── TCP streaming output ─────────────────────────────────────────

    def _add_stream_branch(self, tee: Gst.Element) -> None:
        """Add tee branch: encode H264 → MKV → TCP server.

        Clients connect to: tcp://<host>:<stream_port>
        """
        q_stream = make_queue("q_stream_out")
        nvvidconv_stream = make_element("nvvideoconvert", "nvvidconv_stream")
        nvvidconv_stream.set_property("gpu-id", GPU_ID)

        q_caps_stream = make_queue("q_caps_stream")
        caps_stream = make_element("capsfilter", "caps_stream")
        caps_stream.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        )

        q_enc_stream = make_queue("q_enc_stream")
        encoder_stream = make_element("nvv4l2h264enc", "encoder_stream")
        encoder_stream.set_property("bitrate", 8000000)
        encoder_stream.set_property("iframeinterval", 30)
        encoder_stream.set_property("profile", 4)  # high
        encoder_stream.set_property("insert-sps-pps", True)
        encoder_stream.set_property("insert-vui", True)

        q_parse_stream = make_queue("q_parse_stream")
        h264parse_stream = make_element("h264parse", "h264parse_stream")
        h264parse_stream.set_property("config-interval", -1)

        q_mux_stream = make_queue("q_mux_stream")
        mkvmux = make_element("matroskamux", "mkvmux")
        mkvmux.set_property("streamable", True)

        tcpsink = make_element("tcpserversink", "tcpsink")
        tcpsink.set_property("host", "0.0.0.0")
        tcpsink.set_property("port", self.stream_port)
        tcpsink.set_property("sync", 0)
        tcpsink.set_property("async", 0)

        for el in [
            q_stream, nvvidconv_stream, q_caps_stream, caps_stream,
            q_enc_stream, encoder_stream, q_parse_stream, h264parse_stream,
            q_mux_stream, mkvmux, tcpsink,
        ]:
            self.pipeline.add(el)

        tee.link(q_stream)
        q_stream.link(nvvidconv_stream)
        nvvidconv_stream.link(q_caps_stream)
        q_caps_stream.link(caps_stream)
        caps_stream.link(q_enc_stream)
        q_enc_stream.link(encoder_stream)
        encoder_stream.link(q_parse_stream)
        q_parse_stream.link(h264parse_stream)
        h264parse_stream.link(q_mux_stream)
        q_mux_stream.link(mkvmux)
        mkvmux.link(tcpsink)

        logger.info(
            "TCP stream server on port %d — view with: "
            "python -c \"import cv2; cap=cv2.VideoCapture('tcp://localhost:%d')\"",
            self.stream_port, self.stream_port,
        )

    # ── Run ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start pipeline and block until error or EOS."""
        if self.pipeline is None:
            self.build()

        self.loop = GLib.MainLoop()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to start pipeline")
            return

        if self.record_duration > 0:
            GLib.timeout_add_seconds(
                self.record_duration, self._on_duration_reached
            )
            logger.info("Recording for %d seconds...", self.record_duration)

        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.stop()

    def _on_duration_reached(self) -> bool:
        """Send EOS to finalize files when record duration expires."""
        logger.info("Record duration reached, sending EOS...")
        self.pipeline.send_event(Gst.Event.new_eos())
        return False  # don't repeat

    def stop(self) -> None:
        """Clean shutdown."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        for f in self.jsonl_files.values():
            f.close()
        logger.info("Pipeline stopped.")

    def _on_bus_message(self, bus, message) -> None:
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            logger.info("End of stream")
            self.loop.quit()
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src_name = message.src.get_name() if message.src else "unknown"
            # Don't kill pipeline for rtspsrc errors — camera may reconnect
            if "rtspsrc" in src_name:
                logger.warning("Source %s error (non-fatal): %s", src_name, err)
            else:
                logger.error("Pipeline error: %s\n%s", err, debug)
                self.loop.quit()
        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("Pipeline warning: %s\n%s", err, debug)


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="BPA Vision DeepStream Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sources",
        nargs="+",
        help="RTSP URIs for camera sources",
    )
    group.add_argument(
        "--sources-file",
        help="JSON file with camera sources: [{camera_id, stream_uri}, ...]",
    )
    parser.add_argument(
        "--pose-config",
        default="/app/deepstream/config_infer_yolo26m_pose.yml",
        help="Path to nvinfer pose config",
    )
    parser.add_argument(
        "--output-dir",
        default="/app/output",
        help="Directory for output files",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=0,
        help="TCP stream server port (0 = disabled). View with: "
             "cv2.VideoCapture('tcp://localhost:<port>')",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=60,
        help="MP4 segment duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--record-duration",
        type=int,
        default=0,
        help="Stop pipeline after N seconds (0 = unlimited)",
    )
    parser.add_argument(
        "--calibration-dir",
        default="",
        help="Directory with calibration_<camera_id>.yaml files for undistortion. "
             "If empty or file not found, undistortion is skipped.",
    )
    args = parser.parse_args()

    if args.sources_file:
        with open(args.sources_file, "r", encoding="utf-8") as f:
            sources = json.load(f)
    else:
        sources = []
        for idx, uri in enumerate(args.sources):
            sources.append({
                "camera_id": f"cam_{idx}",
                "stream_uri": uri,
            })

    mgr = PipelineManager(
        sources=sources,
        pose_config=args.pose_config,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        stream_port=args.stream_port,
        segment_duration=args.segment_duration,
        record_duration=args.record_duration,
        calibration_dir=args.calibration_dir,
    )
    mgr.build()
    mgr.run()


if __name__ == "__main__":
    main()
