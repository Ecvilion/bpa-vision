"""Pydantic models for YAML configuration validation.

All coordinates in config must be in FrameNormalizedSpace [0,1].
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from bpa_vision.domain.enums import TrackingProfile, ZoneType


class SamplingConfig(BaseModel):
    """Frame sampling settings."""
    target_fps: float = Field(default=5.0, gt=0)
    max_batch_size: int = Field(default=1, ge=1)


class GeometryConfig(BaseModel):
    """Geometry/calibration settings for a camera."""
    enabled: bool = False
    distortion_correction: bool = False
    homography_points: Optional[list[list[float]]] = None


class TrackingConfig(BaseModel):
    """Tracking parameters."""
    profile: TrackingProfile = TrackingProfile.LITE
    t_lost_seconds: float = Field(default=5.0, gt=0)
    reactivation_enabled: bool = True


class ZoneConfig(BaseModel):
    """Zone definition in YAML."""
    zone_id: str
    name: Optional[str] = None
    zone_type: ZoneType = ZoneType.POLYGON
    points: list[list[float]]  # [[x,y], ...] in [0,1]


class RuleConfig(BaseModel):
    """Analytics rule template + params."""
    rule_id: str
    rule_type: str  # e.g. "line_crossing", "dwell", "zone_presence"
    zone_id: Optional[str] = None
    params: dict = Field(default_factory=dict)


class AnalyticsConfig(BaseModel):
    """Analytics bindings for a camera."""
    rules: list[RuleConfig] = Field(default_factory=list)


class CameraConfig(BaseModel):
    """Camera configuration block."""
    camera_id: str
    stream_uri: str
    codec: str = "h264"
    resolution: list[int] = Field(default=[1920, 1080])
    fps: float = 25.0
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    zones: list[ZoneConfig] = Field(default_factory=list)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)


class IdentityConfig(BaseModel):
    """Identity association policy."""
    body_embedding_dim: int = 256
    face_embedding_dim: int = 512
    retrieve_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    provisional_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    confirm_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    candidate_margin_min: float = Field(default=0.05, ge=0.0)
    k_max_per_view: int = Field(default=3, ge=1)
    face_confirm_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class RetentionConfig(BaseModel):
    """Data retention policy."""
    tracks_days: int = Field(default=30, ge=1)
    events_days: int = Field(default=90, ge=1)
    crops_days: int = Field(default=14, ge=1)
    embeddings_days: int = Field(default=90, ge=1)


class SiteConfig(BaseModel):
    """Top-level site configuration — root of the YAML."""
    site_id: str
    site_name: str
    cameras: list[CameraConfig] = Field(default_factory=list)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
