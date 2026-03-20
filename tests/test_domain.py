"""Tests for domain models — instantiation and validation."""

from datetime import datetime
from uuid import uuid4

import pytest

from bpa_vision.domain.enums import (
    AssociationState,
    HypothesisState,
    PersonResolutionMode,
    TrackState,
    ViewType,
)
from bpa_vision.domain.models import (
    AssociationEdge,
    BodyObservation,
    BodyPrototype,
    DerivedEvent,
    FaceObservation,
    FrameObservation,
    IdentityHypothesis,
    LocalTrack,
    NormalizedBBox,
    NormalizedPoint,
    PrimitiveEvent,
    ResolvedPerson,
    WorldState,
)


class TestNormalizedBBox:
    def test_basic_properties(self):
        bbox = NormalizedBBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8)
        assert bbox.width == pytest.approx(0.4)
        assert bbox.height == pytest.approx(0.6)
        assert bbox.area == pytest.approx(0.24)
        assert bbox.center == pytest.approx((0.3, 0.5))

    def test_rejects_out_of_range(self):
        with pytest.raises(Exception):
            NormalizedBBox(x_min=-0.1, y_min=0.0, x_max=0.5, y_max=0.5)
        with pytest.raises(Exception):
            NormalizedBBox(x_min=0.0, y_min=0.0, x_max=1.5, y_max=0.5)


class TestFrameObservation:
    def test_create_minimal(self):
        obs = FrameObservation(
            camera_id="cam1",
            frame_idx=42,
            timestamp=datetime(2026, 1, 1),
            bbox=NormalizedBBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8),
            detection_confidence=0.9,
        )
        assert obs.camera_id == "cam1"
        assert obs.frame_idx == 42
        assert obs.visibility_score == 1.0
        assert obs.occlusion_score == 0.0
        assert obs.world_point is None

    def test_rejects_invalid_confidence(self):
        with pytest.raises(Exception):
            FrameObservation(
                camera_id="cam1",
                frame_idx=0,
                timestamp=datetime(2026, 1, 1),
                bbox=NormalizedBBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
                detection_confidence=1.5,
            )


class TestLocalTrack:
    def test_defaults(self):
        track = LocalTrack(camera_id="cam1")
        assert track.state == TrackState.NEW
        assert track.association_state == AssociationState.UNLINKED
        assert track.hypothesis_id is None
        assert track.observations == []
        assert track.track_length_frames == 0

    def test_state_values(self):
        track = LocalTrack(camera_id="cam1", state=TrackState.ACTIVE)
        assert track.state == TrackState.ACTIVE


class TestIdentityHypothesis:
    def test_defaults(self):
        hyp = IdentityHypothesis()
        assert hyp.state == HypothesisState.TENTATIVE
        assert hyp.prototypes == []
        assert hyp.support_count == 0

    def test_with_prototype(self):
        proto = BodyPrototype(
            view_type=ViewType.FRONT,
            embedding=[0.1] * 128,
        )
        hyp = IdentityHypothesis(prototypes=[proto])
        assert len(hyp.prototypes) == 1
        assert hyp.prototypes[0].view_type == ViewType.FRONT


class TestResolvedPerson:
    def test_create(self):
        person = ResolvedPerson(
            hypothesis_id=uuid4(),
            resolution_mode=PersonResolutionMode.DIRECT_FACE,
            face_embedding=[0.1] * 512,
        )
        assert person.resolution_mode == PersonResolutionMode.DIRECT_FACE
        assert len(person.face_embedding) == 512


class TestEvents:
    def test_primitive_event(self):
        event = PrimitiveEvent(
            event_type="line_crossing",
            camera_id="cam1",
            timestamp=datetime(2026, 1, 1),
            payload={"direction": "in"},
        )
        assert event.event_type == "line_crossing"
        assert event.payload["direction"] == "in"

    def test_derived_event(self):
        src_id = uuid4()
        event = DerivedEvent(
            event_type="customer_waiting_no_staff",
            source_event_ids=[src_id],
            timestamp=datetime(2026, 1, 1),
        )
        assert len(event.source_event_ids) == 1
