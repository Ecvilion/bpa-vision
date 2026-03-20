"""Canonical domain entities.

All spatial coordinates use FrameNormalizedSpace (x,y in [0,1])
unless explicitly noted otherwise.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .enums import (
    AssociationState,
    EdgeType,
    HypothesisState,
    PersonResolutionMode,
    TrackingProfile,
    TrackState,
    ViewType,
    ZoneType,
)


# ---------------------------------------------------------------------------
# Geometry primitives (FrameNormalizedSpace)
# ---------------------------------------------------------------------------

class NormalizedBBox(BaseModel):
    """Bounding box in FrameNormalizedSpace [0,1]."""
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class NormalizedPoint(BaseModel):
    """Point in FrameNormalizedSpace [0,1]."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class Keypoint(BaseModel):
    """Single pose keypoint in FrameNormalizedSpace."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Infrastructure entities
# ---------------------------------------------------------------------------

class Calibration(BaseModel):
    """Camera calibration parameters."""
    calibration_id: UUID = Field(default_factory=uuid4)
    intrinsics: Optional[list[float]] = None
    distortion_coeffs: Optional[list[float]] = None
    homography_matrix: Optional[list[list[float]]] = None
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_valid: bool = False


class Zone(BaseModel):
    """Geometric region or line inside a camera view (normalized coords)."""
    zone_id: str
    zone_type: ZoneType
    points: list[NormalizedPoint]
    name: Optional[str] = None


class Camera(BaseModel):
    """Video source definition."""
    camera_id: str
    stream_uri: str
    codec: str = "h264"
    resolution: tuple[int, int] = (1920, 1080)
    fps: float = 25.0
    calibration: Optional[Calibration] = None
    zones: list[Zone] = Field(default_factory=list)


class Node(BaseModel):
    """Compute node on site. One primary GPU in v1."""
    node_id: str
    cameras: list[Camera] = Field(default_factory=list)


class Site(BaseModel):
    """Logical deployment: store, factory, hospital, office."""
    site_id: str
    name: str
    nodes: list[Node] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class FrameObservation(BaseModel):
    """Single person observation on one frame.

    bbox and pose are in FrameNormalizedSpace.
    """
    observation_id: UUID = Field(default_factory=uuid4)
    camera_id: str
    frame_idx: int
    timestamp: datetime
    bbox: NormalizedBBox
    keypoints: list[Keypoint] = Field(default_factory=list)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    visibility_score: float = Field(default=1.0, ge=0.0, le=1.0)
    occlusion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    world_point: Optional[tuple[float, float]] = None

    model_config = {"arbitrary_types_allowed": True}


class WorldState(BaseModel):
    """Track state in floor/map coordinates."""
    world_x: float
    world_y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    direction: Optional[float] = None  # radians
    projection_confidence: float = Field(ge=0.0, le=1.0)


class BodyObservation(BaseModel):
    """Quality-gated body sample."""
    observation_id: UUID = Field(default_factory=uuid4)
    track_id: UUID
    timestamp: datetime
    bbox: NormalizedBBox
    view_type: Optional[ViewType] = None
    quality_score: float = Field(ge=0.0, le=1.0)
    occlusion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding: Optional[list[float]] = None
    crop_path: Optional[str] = None


class FaceObservation(BaseModel):
    """Quality-gated face sample."""
    observation_id: UUID = Field(default_factory=uuid4)
    track_id: UUID
    timestamp: datetime
    bbox: NormalizedBBox
    quality_score: float = Field(ge=0.0, le=1.0)
    blur_score: float = Field(default=0.0, ge=0.0)
    embedding: Optional[list[float]] = None
    crop_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class LocalTrack(BaseModel):
    """Local person track within one camera.

    Lifecycle: new → active → lost → closed.
    """
    track_id: UUID = Field(default_factory=uuid4)
    camera_id: str
    state: TrackState = TrackState.NEW
    profile: TrackingProfile = TrackingProfile.LITE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    observations: list[FrameObservation] = Field(default_factory=list)
    body_observations: list[BodyObservation] = Field(default_factory=list)
    face_observations: list[FaceObservation] = Field(default_factory=list)

    world_state: Optional[WorldState] = None
    association_state: AssociationState = AssociationState.UNLINKED
    hypothesis_id: Optional[UUID] = None

    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    track_length_frames: int = 0

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class BodyPrototype(BaseModel):
    """Reference body appearance per view bucket for a hypothesis."""
    view_type: ViewType
    embedding: list[float]
    support_count: int = 1
    last_updated: datetime = Field(default_factory=datetime.now)


class IdentityHypothesis(BaseModel):
    """Identity hypothesis — groups tracks and body prototypes.

    Lifecycle: tentative → active → stale → merged/split/archived.
    """
    hypothesis_id: UUID = Field(default_factory=uuid4)
    state: HypothesisState = HypothesisState.TENTATIVE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    prototypes: list[BodyPrototype] = Field(default_factory=list)
    linked_track_ids: list[UUID] = Field(default_factory=list)
    support_count: int = 0
    person_id: Optional[UUID] = None


class ResolvedPerson(BaseModel):
    """Confirmed person identity — only created after face match."""
    person_id: UUID = Field(default_factory=uuid4)
    hypothesis_id: UUID
    resolution_mode: PersonResolutionMode
    face_embedding: list[float]
    created_at: datetime = Field(default_factory=datetime.now)
    name: Optional[str] = None


class AssociationEdge(BaseModel):
    """Edge in the identity graph."""
    edge_id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    edge_type: EdgeType
    score: float = 0.0
    reason: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class PrimitiveEvent(BaseModel):
    """Event directly from track/zone/line/analytics module."""
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    camera_id: str
    track_id: Optional[UUID] = None
    zone_id: Optional[str] = None
    timestamp: datetime
    payload: dict = Field(default_factory=dict)


class DerivedEvent(BaseModel):
    """Composite event from multiple primitives and/or identity states."""
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    source_event_ids: list[UUID] = Field(default_factory=list)
    timestamp: datetime
    payload: dict = Field(default_factory=dict)
