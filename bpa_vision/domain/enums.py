"""Domain enumerations and constants."""

from enum import Enum


class TrackState(str, Enum):
    """LocalTrack lifecycle states."""
    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    CLOSED = "closed"


class TrackingProfile(str, Enum):
    """Tracking complexity profiles."""
    LITE = "lite"
    STRONG = "strong"


class HypothesisState(str, Enum):
    """IdentityHypothesis lifecycle states."""
    TENTATIVE = "tentative"
    ACTIVE = "active"
    STALE = "stale"
    MERGED = "merged"
    SPLIT = "split"
    ARCHIVED = "archived"


class AssociationState(str, Enum):
    """LocalTrack → IdentityHypothesis link states."""
    UNLINKED = "unlinked"
    CANDIDATE_FOUND = "candidate_found"
    PROVISIONAL_LINK = "provisional_link"
    CONFIRMED_LINK = "confirmed_link"
    CONTRADICTED = "contradicted"


class ViewType(str, Enum):
    """Body observation view bucket."""
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"


class ZoneType(str, Enum):
    """Geometric zone types."""
    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    LINE = "line"


class EdgeType(str, Enum):
    """Identity graph edge types."""
    BELONGS_TO = "belongs_to"
    RESOLVED_AS = "resolved_as"
    CONTRADICTION = "contradiction"
    MERGE = "merge"
    SPLIT = "split"


class PersonResolutionMode(str, Enum):
    """How ResolvedPerson was confirmed."""
    DIRECT_FACE = "direct_face"
    INHERITED_VIA_HYPOTHESIS = "inherited_via_hypothesis"
