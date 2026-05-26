from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DETECTORS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DETECTORS_ROOT.parent
DEFAULT_REGISTRY = DETECTORS_ROOT / "configs" / "detectors.yaml"


@dataclass(frozen=True)
class DetectorConfig:
    name: str
    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise KeyError(f"Detector {self.name} missing required config key {key!r}")
        return self.values[key]


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, DetectorConfig]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    detectors = {}
    for name, cfg in raw.get("detectors", {}).items():
        merged = {**defaults, **cfg}
        merged["name"] = name
        detectors[name] = DetectorConfig(name=name, values=merged)
    return detectors


def resolve_detector_config(name: str, path: str | Path = DEFAULT_REGISTRY) -> DetectorConfig:
    registry = load_registry(path)
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown detector {name!r}. Known detectors: {known}")
    return registry[name]


def vendor_path(cfg: DetectorConfig) -> Path:
    rel = cfg.require("vendor_path")
    path = DETECTORS_ROOT / str(rel)
    if not path.exists():
        raise FileNotFoundError(f"Vendor path for {cfg.name} does not exist: {path}")
    return path
