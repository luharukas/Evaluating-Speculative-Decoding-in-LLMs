"""Main pipeline: load config, run baseline + Eagle3, report metrics."""

import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from .dataset import load_dataset_prompts
from .metrics import compare_outputs, print_comparison_table
from .runners import BaseRunner, Eagle3Runner


load_dotenv()  # Load environment variables from .env file if present


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class EaglePipeline:
    """Orchestrates baseline and Eagle3 evaluation and reports all metrics."""

    def __init__(self, config: dict):
        self.cfg = config
        self.results_dir = Path(config["output"]["save_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_prompts(self) -> list[str]:
        eval_cfg = self.cfg["evaluation"]
        return load_dataset_prompts(
            dataset=eval_cfg["dataset"],
            num_samples=eval_cfg["num_samples"],
            custom_path=eval_cfg.get("custom_prompts_path"),
        )

    def run(
        self,
        run_baseline: bool = True,
        run_eagle: bool = True,
        skip_baseline: bool = False,
    ):
        prompts = self._load_prompts()
        print(f"\n[Pipeline] Loaded {len(prompts)} prompts from '{self.cfg['evaluation']['dataset']}'")

        gen_cfg = self.cfg["generation"]
        compute_cfg = self.cfg["compute"]
        models_cfg = self.cfg["models"]
        spec_cfg = self.cfg["speculative_decoding"]
        warmup = compute_cfg.get("warmup_runs", 3)

        baseline_metrics = None
        eagle_metrics = None

        # ── Baseline Run ──────────────────────────────────────────────────────
        if run_baseline and not skip_baseline:
            print("\n[Pipeline] === BASELINE RUN (no speculative decoding) ===")
            base_runner = BaseRunner(
                model=models_cfg["base_model"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                gpu_memory_utilization=compute_cfg["gpu_memory_utilization"],
                dtype=compute_cfg["dtype"],
            )
            baseline_collector = base_runner.run(prompts, warmup_runs=warmup)
            baseline_metrics = baseline_collector.build()
            # Free GPU memory before loading Eagle3
            del base_runner
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # ── Eagle3 Run ────────────────────────────────────────────────────────
        if run_eagle:
            print("\n[Pipeline] === EAGLE3 SPECULATIVE DECODING RUN ===")
            eagle_runner = Eagle3Runner(
                base_model=models_cfg["base_model"],
                draft_model=models_cfg["eagle3_draft_model"],
                num_speculative_tokens=spec_cfg["num_speculative_tokens"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                gpu_memory_utilization=compute_cfg["gpu_memory_utilization"],
                dtype=compute_cfg["dtype"],
                draft_tensor_parallel_size=spec_cfg.get("draft_tensor_parallel_size", 1),
            )
            eagle_collector = eagle_runner.run(prompts, warmup_runs=warmup)
            eagle_metrics = eagle_collector.build()
            del eagle_runner
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # ── Compare & Report ─────────────────────────────────────────────────
        if baseline_metrics and eagle_metrics:
            eagle_metrics.exact_match_rate = compare_outputs(baseline_metrics, eagle_metrics)
            print_comparison_table(baseline_metrics, eagle_metrics)
            self._save_results(baseline_metrics, eagle_metrics)
        elif baseline_metrics:
            self._print_single_run(baseline_metrics)
            self._save_single(baseline_metrics, "baseline")
        elif eagle_metrics:
            self._print_single_run(eagle_metrics)
            self._save_single(eagle_metrics, "eagle3")

        return baseline_metrics, eagle_metrics

    def _print_single_run(self, m):
        print(f"\n{'='*50}")
        print(f"  {m.run_type.upper()} RESULTS — {m.model}")
        print(f"{'='*50}")
        print(f"  Samples          : {m.num_samples}")
        print(f"  Total Time (s)   : {m.total_time:.1f}")
        print(f"  Mean TTFT (ms)   : {m.mean_ttft_ms:.1f}")
        print(f"  P90 TTFT (ms)    : {m.p90_ttft_ms:.1f}")
        print(f"  Mean E2E (ms)    : {m.mean_e2e_latency_ms:.1f}")
        print(f"  Mean TPS         : {m.mean_tps:.1f}")
        print(f"  Throughput TPS   : {m.throughput_tps:.1f}")
        if m.mean_acceptance_rate is not None:
            print(f"  Acceptance Rate  : {m.mean_acceptance_rate:.3f}")
        if m.mean_accepted_per_step is not None:
            print(f"  Accepted/Step    : {m.mean_accepted_per_step:.2f}")
        print(f"{'='*50}\n")

    def _save_results(self, baseline, eagle):
        ts = int(time.time())
        out = {
            "timestamp": ts,
            "baseline": baseline.to_dict(),
            "eagle3": eagle.to_dict(),
        }
        if self.cfg["output"].get("save_detailed_json"):
            out["baseline"]["per_request"] = baseline.per_request
            out["eagle3"]["per_request"] = eagle.per_request

        path = self.results_dir / f"results_{ts}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[Pipeline] Results saved to {path}")

    def _save_single(self, metrics, tag: str):
        ts = int(time.time())
        out = {tag: metrics.to_dict()}
        if self.cfg["output"].get("save_detailed_json"):
            out[tag]["per_request"] = metrics.per_request
        path = self.results_dir / f"results_{tag}_{ts}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[Pipeline] Results saved to {path}")
