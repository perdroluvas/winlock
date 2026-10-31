"""Loading of the analysis parameters from config/analysis.toml."""

import tomllib
from pathlib import Path
from typing import Any

# src/winlock/config.py -> parents[0]=winlock, [1]=src, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "analysis.toml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the TOML config file and return it as a dictionary."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_path(relative: str) -> Path:
    """Turn a config path (relative to the project root) into an absolute path."""
    return PROJECT_ROOT / relative
