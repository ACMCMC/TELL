"""Evaluate any checkpoint with the shared eval pipeline."""

import argparse
import asyncio
import datetime
import logging
import pathlib

import tinker
import wandb
import weave
from dotenv import load_dotenv
from rl_detector.config import CFG
from rl_detector.prompt_utils import load_tokenizer
from rl_detector.data import load_docs, truncate_documents_in_place
from rl_detector.eval_runner import evaluate_model, select_eval_docs
from rl_detector.train import RunLogger

load_dotenv()
logger = logging.getLogger(__name__)


async def _run(checkpoint: str, max_eval_docs: int, run_name: str | None, eval_docs_path: str | None) -> None:
    if run_name:
        CFG.wandb.name = run_name
    CFG.data.max_eval_docs = int(max_eval_docs)
    if eval_docs_path:
        CFG.data.eval_docs_path = eval_docs_path

    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = (getattr(CFG.wandb, "name", None) or "eval_checkpoint").replace("/", "-").replace(" ", "_")
    run_dir = pathlib.Path("runs") / f"{dt}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run directory: %s", run_dir.resolve())
    eval_audit_path = str(run_dir / "eval_audit_log.jsonl")

    if CFG.wandb.enabled:
        wandb.init(
            project=CFG.wandb.project,
            entity=CFG.wandb.entity,
            name=getattr(CFG.wandb, "name", "eval_checkpoint"),
            config={"checkpoint": checkpoint, "max_eval_docs": int(max_eval_docs)},
            resume="allow",
        )
    weave_enabled = bool(getattr(CFG.wandb, "weave_trace", True))
    if weave_enabled:
        entity = getattr(CFG.wandb, "entity", None) or ""
        proj = f"{entity}/{CFG.wandb.project}" if entity else CFG.wandb.project
        weave.init(project_name=proj, settings={"implicitly_patch_integrations": False})

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_training_client_from_state_with_optimizer_async(path=checkpoint)
    tokenizer = load_tokenizer()

    # load_docs uses data.eval_docs_path; dataset_ids kept only for old call sites
    test_docs = load_docs(None, use_eval_split=True, max_docs=int(CFG.data.max_eval_docs))
    n_short = truncate_documents_in_place(tokenizer=tokenizer, docs=test_docs, max_doc_tokens=int(CFG.data.max_doc_tokens))
    logger.info("eval docs loaded=%d truncated=%d", len(test_docs), n_short)
    eval_docs = select_eval_docs(
        test_docs,
        sample_size=int(CFG.data.max_eval_docs),
        seed=int(getattr(CFG.frozen, "seed", 2262)),
    )

    metrics = await evaluate_model(
        training_client=training_client,
        tokenizer=tokenizer,
        eval_docs=eval_docs,
        step="eval-checkpoint",
        eval_seed=int(getattr(CFG.frozen, "seed", 2262)),
        eval_audit_path=eval_audit_path,
    )
    logger.info("eval-only | AUROC=%.4f TPR@1%%FPR=%.4f format_rate=%.3f reward=%.4f", metrics["eval_auroc"], metrics["eval_tpr_at_fpr_001"], metrics["eval_format_rate"], metrics["eval_reward_mean"])
    logger.info("eval audit log: %s", eval_audit_path)

    stratum_stats = metrics.get("stratum_stats", {})
    if stratum_stats:
        lines = [f"{'stratum':<55} {'n':>4} {'reward':>8} {'agg_score':>10} {'format':>8} {'auroc':>7}"]
        for k, v in sorted(stratum_stats.items(), key=lambda x: x[1].get("reward_mean", 0.0)):
            auroc_str = f"{v['auroc']:.4f}" if v["auroc"] is not None else "   n/a"
            lines.append(f"  {k:<55} {v['n']:>4} {v['reward_mean']:>8.4f} {v['agg_score_mean']:>10.4f} {v['format_rate']:>8.3f} {auroc_str:>7}")
        print("\n=== PER-STRATUM BREAKDOWN (sorted by reward_mean asc) ===")
        print("\n".join(lines))
        print("=" * 80)

    if CFG.wandb.enabled:
        run_logger = RunLogger(str(run_dir / "metrics_log.jsonl"))
        run_logger.log_eval(metrics, step=0)
        for k, v in stratum_stats.items():
            ds, dom = k.rsplit("|", 1)
            prefix = f"eval_stratum/{ds}/{dom}"
            run_logger.log({
                f"{prefix}/reward_mean": v["reward_mean"],
                f"{prefix}/agg_score_mean": v["agg_score_mean"],
                f"{prefix}/format_rate": v["format_rate"],
                **({f"{prefix}/auroc": v["auroc"]} if v["auroc"] is not None else {}),
            }, step=0)
        run_logger.close()
        wandb.finish()
    if weave_enabled:
        weave.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with shared RL eval.")
    parser.add_argument("--checkpoint", required=True, help="tinker://... checkpoint path")
    parser.add_argument("--max-eval-docs", type=int, default=50)
    parser.add_argument("--name", default=None, help="Optional wandb run name override")
    parser.add_argument("--eval-docs-path", default=None, help="Override data.eval_docs_path (e.g. hf://acmc/multi_domain_ai_human_text/test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_run(checkpoint=args.checkpoint, max_eval_docs=args.max_eval_docs, run_name=args.name, eval_docs_path=args.eval_docs_path))


if __name__ == "__main__":
    main()
