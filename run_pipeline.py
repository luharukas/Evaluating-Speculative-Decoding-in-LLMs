#!/usr/bin/env python3
"""
Inference Backend Evaluation Pipeline — vLLM vs TensorRT-LLM

Usage:
    python run_pipeline.py                           # both backends, all 4 runs
    python run_pipeline.py --backend vllm            # vLLM only (baseline + Eagle3)
    python run_pipeline.py --backend trtllm          # TRT-LLM only
    python run_pipeline.py --backend both            # all 4 runs (default)
    python run_pipeline.py --baseline-only           # baselines only (no Eagle3)
    python run_pipeline.py --eagle-only              # Eagle3 only (no baselines)
    python run_pipeline.py --config my_config.yaml
    python run_pipeline.py --base-model <hf_id> --eagle3-model <hf_id>
    python run_pipeline.py --num-samples 10          # quick smoke test
    python run_pipeline.py --prompts path/to/prompts.jsonl
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import EaglePipeline, load_config

load_dotenv()


def ensure_interpreter_bin_on_path():
    """Prepend the current interpreter's bin dir so subprocess tools resolve.

    This matters when the script is launched as:
      .venv/bin/python run_pipeline.py ...
    without activating the virtualenv first. In that mode, Python packages are
    imported from the venv correctly, but helper executables installed into the
    same venv (for example `ninja`) are not on PATH unless we add them.
    """
    bindir = str(Path(sys.executable).parent.resolve())
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if bindir not in parts:
        os.environ["PATH"] = bindir + (os.pathsep + path if path else "")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate speculative decoding across vLLM and TRT-LLM backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    p.add_argument("--backend", choices=["vllm", "trtllm", "both"], default="both",
                   help="Inference backend(s) to run (default: both)")
    p.add_argument("--base-model",  help="Override base model HuggingFace ID or local path")
    p.add_argument("--eagle3-model", help="Override Eagle3 draft model HuggingFace ID or local path")
    p.add_argument("--num-samples", type=int, help="Number of prompts to evaluate")
    p.add_argument("--max-new-tokens", type=int, help="Max tokens to generate per prompt")
    p.add_argument("--num-speculative-tokens", type=int, help="Draft tokens per speculative step")
    p.add_argument("--dataset", choices=["sharegpt", "mt_bench", "custom"],
                   help="Evaluation dataset")
    p.add_argument("--prompts", help="Path to custom JSONL prompts file")
    p.add_argument("--gpu-util", default=0.5, type=float, help="GPU memory utilization (0–1)")
    p.add_argument("--warmup-runs", type=int, help="Warmup runs before measurement")
    p.add_argument("--results-dir", help="Directory to save results JSON")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run baseline variants only (skip Eagle3)")
    p.add_argument("--eagle-only", action="store_true",
                   help="Run Eagle3 variants only (skip baselines)")
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
    ensure_interpreter_bin_on_path()
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    backend = args.backend
    want_baseline = not args.eagle_only
    want_eagle    = not args.baseline_only

    run_vllm_baseline    = backend in ("vllm", "both")  and want_baseline
    run_vllm_eagle       = backend in ("vllm", "both")  and want_eagle
    run_trtllm_baseline  = backend in ("trtllm", "both") and want_baseline
    run_trtllm_eagle     = backend in ("trtllm", "both") and want_eagle

    print("\n Inference Backend Evaluation Pipeline")
    print(f"  Backend     : {backend}")
    print(f"  Base model  : {cfg['models']['base_model']}")
    if run_vllm_eagle or run_trtllm_eagle:
        print(f"  Eagle3 draft: {cfg['models']['eagle3_draft_model']}")
    print(f"  Dataset     : {cfg['evaluation']['dataset']} "
          f"({cfg['evaluation']['num_samples']} samples)")
    print(f"  Max tokens  : {cfg['generation']['max_new_tokens']}")
    if run_vllm_eagle or run_trtllm_eagle:
        print(f"  Spec tokens : {cfg['speculative_decoding']['num_speculative_tokens']}")
    print(f"  Runs        : "
          + ", ".join(filter(None, [
              "vLLM-baseline"   if run_vllm_baseline   else None,
              "vLLM-Eagle3"     if run_vllm_eagle       else None,
              "TRT-LLM-baseline" if run_trtllm_baseline else None,
              "TRT-LLM-Eagle3"  if run_trtllm_eagle     else None,
          ])))
    print()

    pipeline = EaglePipeline(cfg)
    pipeline.run(
        run_vllm_baseline=run_vllm_baseline,
        run_vllm_eagle=run_vllm_eagle,
        run_trtllm_baseline=run_trtllm_baseline,
        run_trtllm_eagle=run_trtllm_eagle,
    )


if __name__ == "__main__":
    main()
