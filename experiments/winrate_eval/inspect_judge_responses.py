"""Print raw judge API outputs from cache; flags fake A1..An fallback orders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from winrate_judges import JUDGE_PANEL


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/winrate_eval/judge_rankings/suraj-ranganath_tell-human-detectors_validation/d968bd23749d",
    )
    parser.add_argument("--doc-id", type=str, default="")
    args = parser.parse_args()
    base = Path(args.cache_dir)
    for judge_dir in sorted(base.iterdir()):
        if not judge_dir.is_dir():
            continue
        judge_id = judge_dir.name
        paths = sorted(judge_dir.glob("*.json"), key=lambda p: int(p.stem))
        if args.doc_id:
            paths = [p for p in paths if p.stem == args.doc_id]
        print(f"\n{'=' * 70}\n{judge_id}\n{'=' * 70}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            jr = payload["judge_result"]
            raw = jr.get("raw_content", "")
            ranking = jr.get("parsed", {}).get("ranking", [])
            ids = [r.get("item_id") for r in ranking]
            n = len(ids)
            degenerate = ids == [f"A{i}" for i in range(1, n + 1)]
            print(f"doc {path.stem}: cache_key={payload.get('cache_key')} structured={jr.get('structured_parse')}")
            print(f"  degenerate_fallback_order={degenerate} n_rank={n} win_rate={jr.get('win_stats', {}).get('win_rate')}")
            print(f"  raw_len={len(raw)}")
            if len(raw) <= 1200:
                print(f"  RAW:\n{raw}")
            else:
                print(f"  RAW head:\n{raw[:600]}\n  ...\n  RAW tail:\n{raw[-400:]}")


if __name__ == "__main__":
    main()
