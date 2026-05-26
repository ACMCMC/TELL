"""Load Hydra `conf/config.yaml`, expose a DictConfig as `CFG`."""

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# sampling cap tied to doc prefix; used in yaml as ${mul:4,${data.max_doc_tokens}}
if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b))

_CONF_DIR = (Path(__file__).resolve().parent.parent.parent / "conf").as_posix()
_CONFIG_NAME = "config"


def load_config() -> DictConfig:
    with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        cfg = compose(config_name=_CONFIG_NAME, overrides=[])
    return cfg


def config_as_dict(cfg: DictConfig) -> dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return out


CFG: DictConfig = load_config()
