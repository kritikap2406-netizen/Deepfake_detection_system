"""
TrueSight — shared config loader.

Every module that needs a threshold, path, or contract field defined in
configs/pipeline.yaml should go through this loader instead of
hardcoding the value or re-parsing the YAML itself. Cached after first
load since the config doesn't change during a process's lifetime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "pipeline.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)
