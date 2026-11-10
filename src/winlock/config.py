import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "analysis.toml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_path(relative: str) -> Path:
    return PROJECT_ROOT / relative
