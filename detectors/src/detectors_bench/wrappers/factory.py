from __future__ import annotations

from detectors_bench.registry import DetectorConfig

from .base import DetectorWrapper
from .binoculars import BinocularsWrapper
from .dnagpt import DNAGPTWrapper
from .fast_detectgpt import FastDetectGPTWrapper
from .ghostbuster import GhostbusterWrapper
from .hf_classifier import HFClassifierWrapper
from .lm_features import DetectLLMLRRWrapper, LogRankWrapper
from .mage import MageWrapper
from .meld import MELDWrapper
from .mfd import MFDWrapper
from .pangram_editlens import PangramEditLensWrapper
from .perturbation import PerturbationMetricWrapper
from .phd import PHDWrapper
from .t5_sentinel import T5SentinelWrapper


def make_wrapper(cfg: DetectorConfig) -> DetectorWrapper:
    kind = cfg.require("kind")
    if kind == "hf_classifier":
        return HFClassifierWrapper(cfg)
    if kind == "mage":
        return MageWrapper(cfg)
    if kind == "binoculars":
        return BinocularsWrapper(cfg)
    if kind == "fast_detectgpt":
        return FastDetectGPTWrapper(cfg)
    if kind == "metric_lm" and cfg.get("method") == "lrr":
        return DetectLLMLRRWrapper(cfg)
    if kind == "metric_lm" and cfg.get("method") == "logrank":
        return LogRankWrapper(cfg)
    if kind == "perturbation_metric":
        return PerturbationMetricWrapper(cfg)
    if kind == "phd":
        return PHDWrapper(cfg)
    if kind == "t5_sentinel":
        return T5SentinelWrapper(cfg)
    if kind == "meld":
        return MELDWrapper(cfg)
    if kind == "dnagpt":
        return DNAGPTWrapper(cfg)
    if kind == "mfd":
        return MFDWrapper(cfg)
    if kind == "ghostbuster":
        return GhostbusterWrapper(cfg)
    if kind == "pangram_editlens":
        return PangramEditLensWrapper(cfg)
    if kind == "gpt_oss_direct":
        from .gpt_oss_direct import GptOssDirectWrapper

        return GptOssDirectWrapper(cfg)
    raise ValueError(f"Unsupported detector kind for {cfg.name}: {kind}")
