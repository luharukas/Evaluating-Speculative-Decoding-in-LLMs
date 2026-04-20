#!/usr/bin/env python3
"""
Eagle3 Speculative Decoding Evaluation Pipeline

Usage:
    python run_pipeline.py                          # use config.yaml defaults
    python run_pipeline.py --config my_config.yaml
    python run_pipeline.py --base-model <hf_id> --eagle3-model <hf_id>
    python run_pipeline.py --eagle-only             # skip baseline, run Eagle3 only
    python run_pipeline.py --baseline-only          # run baseline only
    python run_pipeline.py --num-samples 50         # override sample count
    python run_pipeline.py --prompts path/to/prompts.jsonl  # custom prompts
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import EaglePipeline, load_config
load_dotenv()  # Load environment variables from .env file if present

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Eagle3 speculative decoding vs baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    p.add_argument("--base-model", help="Override base model HuggingFace ID or local path")
    p.add_argument("--eagle3-model", help="Override Eagle3 draft model HuggingFace ID or local path")
    p.add_argument("--num-samples", type=int, help="Number of prompts to evaluate")
    p.add_argument("--max-new-tokens", type=int, help="Max tokens to generate per prompt")
    p.add_argument("--num-speculative-tokens", type=int, help="Number of draft tokens per step")
    p.add_argument("--dataset", choices=["sharegpt", "mt_bench", "custom"],
                   help="Evaluation dataset")
    p.add_argument("--prompts", help="Path to custom JSONL prompts file")
    p.add_argument("--gpu-util", type=float, help="GPU memory utilization (0–1)")
    p.add_argument("--warmup-runs", type=int, help="Number of warmup runs before measurement")
    p.add_argument("--results-dir", help="Directory to save results JSON")
    p.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    p.add_argument("--eagle-only", action="store_true", help="Run Eagle3 only (skip baseline)")
    p.add_argument("--no-save", action="store_true", help="Skip saving results to disk")
    return p.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    if args.base_model:
        cfg["models"]["base_model"] = args.base_model
    if args.eagle3_model:
        cfg["models"]["eagle3_draft_model"] = args.eagle3_model
    if args.num_samples:
        cfg["evaluation"]["num_samples"] = args.num_samples
    if args.max_new_tokens:
        cfg["generation"]["max_new_tokens"] = args.max_new_tokens
    if args.num_speculative_tokens:
        cfg["speculative_decoding"]["num_speculative_tokens"] = args.num_speculative_tokens
    if args.dataset:
        cfg["evaluation"]["dataset"] = args.dataset
    if args.prompts:
        cfg["evaluation"]["dataset"] = "custom"
        cfg["evaluation"]["custom_prompts_path"] = args.prompts
    if args.gpu_util:
        cfg["compute"]["gpu_memory_utilization"] = args.gpu_util
    if args.warmup_runs is not None:
        cfg["compute"]["warmup_runs"] = args.warmup_runs
    if args.results_dir:
        cfg["output"]["save_dir"] = args.results_dir
    if args.no_save:
        cfg["output"]["save_detailed_json"] = False
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    run_baseline = not args.eagle_only
    run_eagle = not args.baseline_only

    print("\n Eagle3 Speculative Decoding Evaluation Pipeline")
    print(f"  Base model  : {cfg['models']['base_model']}")
    if run_eagle:
        print(f"  Eagle3 draft: {cfg['models']['eagle3_draft_model']}")
    print(f"  Dataset     : {cfg['evaluation']['dataset']} "
          f"({cfg['evaluation']['num_samples']} samples)")
    print(f"  Max tokens  : {cfg['generation']['max_new_tokens']}")
    if run_eagle:
        print(f"  Spec tokens : {cfg['speculative_decoding']['num_speculative_tokens']}")
    print()

    pipeline = EaglePipeline(cfg)
    pipeline.run(run_baseline=run_baseline, run_eagle=run_eagle)


if __name__ == "__main__":
    main()
