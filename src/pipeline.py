"""Main pipeline: load config, run all backends, report metrics."""

import json
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from .dataset import load_dataset_prompts
from .metrics import RunMetrics, compare_outputs, print_comparison_table, print_multi_run_table
from .runners import (
    BaseRunner,
    Eagle3Runner,
    TRTLLMBaseRunner,
    TRTLLMEagle3Runner,
)


load_dotenv()

# Ordered keys used consistently for JSON output and table ordering
_RUN_KEYS = ["vllm_baseline", "vllm_eagle3", "trtllm_baseline", "trtllm_eagle3"]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _free_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


class EaglePipeline:
    """Orchestrates baseline and Eagle3 evaluation across vLLM and TRT-LLM."""

    def __init__(self, config: dict):
        self.cfg = config
        self.results_dir = Path(config["output"]["save_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_prompts(self) -> list:
        eval_cfg = self.cfg["evaluation"]
        return load_dataset_prompts(
            dataset=eval_cfg["dataset"],
            num_samples=eval_cfg["num_samples"],
            custom_path=eval_cfg.get("custom_prompts_path"),
        )

    def run(
        self,
        run_vllm_baseline: bool = True,
        run_vllm_eagle: bool = True,
        run_trtllm_baseline: bool = False,
        run_trtllm_eagle: bool = False,
    ) -> dict:
        prompts = self._load_prompts()
        print(f"\n[Pipeline] Loaded {len(prompts)} prompts from '{self.cfg['evaluation']['dataset']}'")

        gen_cfg     = self.cfg["generation"]
        compute_cfg = self.cfg["compute"]
        models_cfg  = self.cfg["models"]
        spec_cfg    = self.cfg["speculative_decoding"]
        warmup      = compute_cfg.get("warmup_runs", 3)

        # TRT-LLM may need a different GPU utilization (managed differently from vLLM)
        trt_gpu_util = (
            self.cfg.get("tensorrt_llm", {}).get("gpu_memory_utilization")
            or compute_cfg["gpu_memory_utilization"]
        )

        results: dict[str, Optional[RunMetrics]] = {k: None for k in _RUN_KEYS}

        # ── vLLM Baseline ─────────────────────────────────────────────────────
        if run_vllm_baseline:
            print("\n[Pipeline] === vLLM BASELINE ===")
            runner = BaseRunner(
                model=models_cfg["base_model"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                gpu_memory_utilization=compute_cfg["gpu_memory_utilization"],
                dtype=compute_cfg["dtype"],
            )
            results["vllm_baseline"] = runner.run(prompts, warmup_runs=warmup).build()
            del runner
            _free_gpu()

        # ── vLLM Eagle3 ───────────────────────────────────────────────────────
        if run_vllm_eagle:
            print("\n[Pipeline] === vLLM EAGLE3 ===")
            runner = Eagle3Runner(
                base_model=models_cfg["base_model"],
                draft_model=models_cfg["eagle3_draft_model"],
                num_speculative_tokens=spec_cfg["num_speculative_tokens"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                gpu_memory_utilization=compute_cfg["gpu_memory_utilization"],
                dtype=compute_cfg["dtype"],
                draft_tensor_parallel_size=spec_cfg.get("draft_tensor_parallel_size", 1),
            )
            results["vllm_eagle3"] = runner.run(prompts, warmup_runs=warmup).build()
            del runner
            _free_gpu()

        # ── TRT-LLM Baseline ──────────────────────────────────────────────────
        if run_trtllm_baseline:
            print("\n[Pipeline] === TRT-LLM BASELINE ===")
            try:
                runner = TRTLLMBaseRunner(
                    model=models_cfg["base_model"],
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    temperature=gen_cfg["temperature"],
                    gpu_memory_utilization=trt_gpu_util,
                    dtype=compute_cfg["dtype"],
                )
                results["trtllm_baseline"] = runner.run(prompts, warmup_runs=warmup).build()
                runner.shutdown()
                del runner
                _free_gpu()
            except ImportError as e:
                print(f"[Pipeline] Skipping TRT-LLM baseline — {e}")

        # ── TRT-LLM Eagle3 ────────────────────────────────────────────────────
        if run_trtllm_eagle:
            print("\n[Pipeline] === TRT-LLM EAGLE3 ===")
            try:
                runner = TRTLLMEagle3Runner(
                    base_model=models_cfg["base_model"],
                    draft_model=models_cfg["eagle3_draft_model"],
                    num_speculative_tokens=spec_cfg["num_speculative_tokens"],
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    temperature=gen_cfg["temperature"],
                    gpu_memory_utilization=trt_gpu_util,
                    dtype=compute_cfg["dtype"],
                )
                results["trtllm_eagle3"] = runner.run(prompts, warmup_runs=warmup).build()
                runner.shutdown()
                del runner
                _free_gpu()
            except ImportError as e:
                print(f"[Pipeline] Skipping TRT-LLM Eagle3 — {e}")

        # ── Compare & Report ──────────────────────────────────────────────────
        present = {k: v for k, v in results.items() if v is not None}
        if not present:
            print("[Pipeline] No runs completed.")
            return results

        # Compute exact-match rate for every run vs vLLM baseline (or first run)
        ref = present.get("vllm_baseline") or next(iter(present.values()))
        for key, m in present.items():
            if m is not ref:
                m.exact_match_rate = compare_outputs(ref, m)

        runs_ordered = [present[k] for k in _RUN_KEYS if k in present]

        if len(runs_ordered) >= 3:
            print_multi_run_table(runs_ordered)
        elif len(runs_ordered) == 2:
            a, b = runs_ordered
            # Use legacy 2-way table only for the original vllm pair
            if {a.run_type, b.run_type} == {"baseline", "eagle3"}:
                print_comparison_table(a, b)
            else:
                print_multi_run_table(runs_ordered)
        else:
            self._print_single_run(runs_ordered[0])

        if self.cfg["output"].get("save_detailed_json", True):
            self._save_results(results)

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _print_single_run(self, m: RunMetrics):
        print(f"\n{'='*50}")
        print(f"  {m.run_type.upper()} — {m.model}")
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

    def _save_results(self, all_results: dict):
        ts = int(time.time())
        save_detailed = self.cfg["output"].get("save_detailed_json", True)

        out: dict = {"timestamp": ts}
        for key, m in all_results.items():
            if m is None:
                out[key] = None
                continue
            out[key] = m.to_dict()
            if save_detailed:
                out[key]["per_request"] = m.per_request

        path = self.results_dir / f"results_{ts}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[Pipeline] Results saved to {path}")
