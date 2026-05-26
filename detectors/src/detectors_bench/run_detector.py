from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from . import __version__
from .io import load_examples, write_json, write_jsonl
from .registry import DEFAULT_REGISTRY, resolve_detector_config
from .schemas import Prediction, attach_example_metadata
from .wrappers import make_wrapper


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return ""


def _manifest_path(output: Path) -> Path:
    suffix = "".join(output.suffixes)
    if suffix:
        return output.with_name(output.name[: -len(suffix)] + ".manifest.json")
    return output.with_name(output.name + ".manifest.json")


def build_manifest(detector: str, cfg: dict, input_path: Path, output_path: Path) -> dict:
    detectors_root = Path(__file__).resolve().parents[2]
    project_root = detectors_root.parent
    return {
        "harness_version": __version__,
        "detector": detector,
        "config": cfg,
        "input": str(input_path),
        "output": str(output_path),
        "python": sys.version,
        "platform": platform.platform(),
        "project_git_sha": _git(["rev-parse", "HEAD"], project_root),
        "detectors_git_status": _git(["status", "--short", "--", "detectors"], project_root),
        "submodule_status": _git(["submodule", "status"], project_root).splitlines(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one detector over a JSONL benchmark file.")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Allow detectors marked enabled=false in the registry. Use only for deliberate optional/appendix runs.",
    )
    parser.add_argument(
        "--allow-init-error",
        action="store_true",
        help="Write one error prediction per example if the detector cannot initialize.",
    )
    args = parser.parse_args(argv)

    examples = load_examples(args.input)
    if args.limit is not None:
        examples = examples[: args.limit]
    cfg = resolve_detector_config(args.detector, args.registry)
    if not cfg.get("enabled", True) and not args.include_disabled:
        reason = cfg.get("exclude_reason", "detector is disabled in the registry")
        raise SystemExit(f"Detector {args.detector!r} is disabled: {reason}")
    try:
        wrapper = make_wrapper(cfg)
        predictions = wrapper.predict_batch(examples)
    except Exception as exc:  # noqa: BLE001
        if not args.allow_init_error:
            raise
        predictions = [
            attach_example_metadata(
                Prediction(id=ex.id, detector=args.detector, score_ai=None, error=f"init_error: {exc!r}"),
                ex,
            )
            for ex in examples
        ]
    write_jsonl(args.output, [p.to_json() for p in predictions])
    write_json(_manifest_path(args.output), build_manifest(args.detector, cfg.values, args.input, args.output))


if __name__ == "__main__":
    main()
