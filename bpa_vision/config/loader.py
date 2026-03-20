"""YAML config loader with pydantic validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import SiteConfig


def load_config(path: str | Path) -> SiteConfig:
    """Load and validate a YAML site config.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated SiteConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML is malformed.
        pydantic.ValidationError: If config doesn't match schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping at top level, got {type(raw).__name__}")

    return SiteConfig(**raw)
