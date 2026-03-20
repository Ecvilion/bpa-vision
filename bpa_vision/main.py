"""BPA Vision entry point — bootstrap runtime from YAML config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bpa_vision import __version__
from bpa_vision.config import load_config

logger = logging.getLogger("bpa_vision")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bpa-vision",
        description="BPA Vision — universal video analytics platform",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML site configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"bpa-vision {__version__}",
    )

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    logger.info("BPA Vision v%s starting", __version__)

    # Load and validate config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    logger.info("Site: %s (%s)", config.site_name, config.site_id)
    logger.info("Cameras: %d", len(config.cameras))

    for cam in config.cameras:
        logger.info(
            "  %s: %s, tracking=%s, zones=%d, rules=%d",
            cam.camera_id,
            cam.stream_uri,
            cam.tracking.profile.value,
            len(cam.zones),
            len(cam.analytics.rules),
        )

    logger.info("Bootstrap complete. Runtime not yet implemented (Stage 1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
