"""vLLM-based runners for baseline and Eagle3 speculative decoding.

vLLM 0.19.1 API notes:
- Speculative decoding uses `speculative_config` dict (not `speculative_model`)
- `RequestOutput.metrics` (RequestStateStats) provides built-in TTFT and timestamps
- Spec decode acceptance stats are accessed via the engine's stat logger
"""

import logging
import re
import time
from typing import Optional

from .metrics import MetricsCollector, RequestMetrics


# ── Log capture for spec decode stats ────────────────────────────────────────

class _SpecDecodeLogCapture(logging.Handler):
    """Captures vLLM's periodic spec decode log lines."""

    PATTERN = re.compile(
        r"Mean acceptance length:\s*([\d.]+).*?"
        r"Accepted:\s*(\d+) tokens.*?"
        r"Drafted:\s*(\d+) tokens",
        re.DOTALL,
    )

    def __init__(self):
        super().__init__()
        self.captured: list[dict] = []

    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        if "SpecDecoding metrics" in msg:
            m = self.PATTERN.search(msg)
            if m:
                self.captured.append({
                    "mean_acceptance_length": float(m.group(1)),
                    "num_accepted": int(m.group(2)),
                    "num_drafted": int(m.group(3)),
                })

    def aggregate(self) -> Optional[dict]:
        if not self.captured:
            return None
        total_accepted = sum(c["num_accepted"] for c in self.captured)
        total_drafted = sum(c["num_drafted"] for c in self.captured)
        # Mean acceptance length (includes bonus token per step)
        mean_mal = sum(c["mean_acceptance_length"] for c in self.captured) / len(self.captured)
        return {
            "acceptance_rate": total_accepted / total_drafted if total_drafted else None,
            "mean_acceptance_length": mean_mal,  # ≈ avg accepted + 1 per step
            "total_accepted": total_accepted,
            "total_drafted": total_drafted,
        }

    def reset(self):
        self.captured.clear()


def _attach_log_capture() -> _SpecDecodeLogCapture:
    handler = _SpecDecodeLogCapture()
    handler.setLevel(logging.DEBUG)
    # vLLM logs to 'vllm' logger hierarchy
    for logger_name in ("vllm", "vllm.v1.metrics.loggers", ""):
        lg = logging.getLogger(logger_name)
        lg.addHandler(handler)
    return handler


def _detach_log_capture(handler: _SpecDecodeLogCapture):
    for logger_name in ("vllm", "vllm.v1.metrics.loggers", ""):
        lg = logging.getLogger(logger_name)
        try:
            lg.removeHandler(handler)
        except Exception:
            pass


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _extract_timings(output) -> tuple[float, float]:
    """Extract TTFT and E2E latency (in seconds) from RequestOutput.metrics."""
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return 0.0, 0.0

    # first_token_latency: seconds from scheduled to first token (vLLM built-in)
    ttft = getattr(metrics, "first_token_latency", 0.0) or 0.0

    # E2E: scheduled → last token
    scheduled_ts = getattr(metrics, "scheduled_ts", 0.0) or 0.0
    last_token_ts = getattr(metrics, "last_token_ts", 0.0) or 0.0
    e2e = (last_token_ts - scheduled_ts) if (last_token_ts and scheduled_ts) else 0.0

    # Fallback: if timestamps are not populated use first_token_latency as TTFT
    # and rely on external wall-clock timing
    return ttft, e2e


# ── Runners ──────────────────────────────────────────────────────────────────

class BaseRunner:
    """Baseline runner — plain LLM, no speculative decoding."""

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.45,
        dtype: str = "float16",
    ):
        from vllm import LLM, SamplingParams
        print(f"[BaseRunner] Loading {model} ...")
        self.model = model
        self.llm = LLM(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,
            disable_log_stats=False,    # enable stats so TTFT is populated
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
        )
        print("[BaseRunner] Model loaded.")

    def warmup(self, prompts: list[str], n: int = 3):
        print(f"[BaseRunner] Warming up ({n} runs) ...")
        for _ in range(n):
            self.llm.generate(prompts[:1], self.sampling_params, use_tqdm=False)

    def run(self, prompts: list[str], warmup_runs: int = 3) -> MetricsCollector:
        self.warmup(prompts, warmup_runs)
        collector = MetricsCollector(run_type="baseline", model=self.model)
        print(f"[BaseRunner] Running {len(prompts)} prompts ...")

        collector.start_run()
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            wall = time.perf_counter() - t0

            out = outputs[0]
            comp = out.outputs[0]
            output_tokens = len(comp.token_ids)
            prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids else 0

            ttft, e2e = _extract_timings(out)
            if e2e <= 0:
                e2e = wall
            if ttft <= 0:
                # Rough estimate: first-token overhead ≈ prefill time (wall / output_tokens)
                ttft = e2e / max(output_tokens, 1)

            tps = output_tokens / e2e if e2e > 0 else 0.0

            collector.add_request(RequestMetrics(
                prompt=prompt,
                output_text=comp.text,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft=ttft,
                e2e_latency=e2e,
                tps=tps,
            ))
            if (i + 1) % 10 == 0:
                elapsed = [r.e2e_latency for r in collector._requests]
                avg_tps = sum(r.tps for r in collector._requests) / len(collector._requests)
                print(f"  [{i+1}/{len(prompts)}] avg TPS: {avg_tps:.1f}")

        collector.end_run()
        return collector


class Eagle3Runner:
    """Eagle3 speculative decoding runner using vLLM 0.19.1 speculative_config API."""

    def __init__(
        self,
        base_model: str,
        draft_model: str,
        num_speculative_tokens: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.45,
        dtype: str = "float16",
        draft_tensor_parallel_size: int = 1,
    ):
        from vllm import LLM, SamplingParams
        print(f"[Eagle3Runner] Loading {base_model}")
        print(f"[Eagle3Runner] + Eagle3 draft: {draft_model}")
        self.base_model = base_model
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens

        self.llm = LLM(
            model=base_model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,
            disable_log_stats=False,
            # Eagle3 speculative decoding config (vLLM 0.19.1+)
            speculative_config={
                "method": "eagle3",
                "model": draft_model,
                "num_speculative_tokens": num_speculative_tokens,
                "draft_tensor_parallel_size": draft_tensor_parallel_size,
            },
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
        )
        print("[Eagle3Runner] Base + Eagle3 draft loaded.")

    def warmup(self, prompts: list[str], n: int = 3):
        print(f"[Eagle3Runner] Warming up ({n} runs) ...")
        for _ in range(n):
            self.llm.generate(prompts[:1], self.sampling_params, use_tqdm=False)

    def run(self, prompts: list[str], warmup_runs: int = 3) -> MetricsCollector:
        self.warmup(prompts, warmup_runs)
        log_handler = _attach_log_capture()

        collector = MetricsCollector(
            run_type="eagle3",
            model=self.base_model,
            draft_model=self.draft_model,
        )
        print(f"[Eagle3Runner] Running {len(prompts)} prompts with Eagle3 ...")

        collector.start_run()
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            wall = time.perf_counter() - t0

            out = outputs[0]
            comp = out.outputs[0]
            output_tokens = len(comp.token_ids)
            prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids else 0

            ttft, e2e = _extract_timings(out)
            if e2e <= 0:
                e2e = wall
            if ttft <= 0:
                ttft = e2e / max(output_tokens, 1)

            tps = output_tokens / e2e if e2e > 0 else 0.0

            # Per-request acceptance rate: estimated from draft steps
            # num_draft_steps ≈ ceil(output_tokens / (num_spec_tokens + 1))
            # Each step: draft K tokens, accept α*K, plus 1 bonus token
            k = self.num_speculative_tokens
            num_draft_steps = max(1, (output_tokens + k) // (k + 1))
            # Estimate accepted tokens = output_tokens - num_draft_steps (bonus tokens)
            estimated_accepted = max(0, output_tokens - num_draft_steps)
            total_drafted = num_draft_steps * k
            acceptance_rate = estimated_accepted / total_drafted if total_drafted > 0 else None

            collector.add_request(RequestMetrics(
                prompt=prompt,
                output_text=comp.text,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft=ttft,
                e2e_latency=e2e,
                tps=tps,
                num_draft_steps=num_draft_steps,
                accepted_tokens=estimated_accepted,
                acceptance_rate=acceptance_rate,
            ))

            if (i + 1) % 10 == 0:
                avg_tps = sum(r.tps for r in collector._requests) / len(collector._requests)
                print(f"  [{i+1}/{len(prompts)}] avg TPS: {avg_tps:.1f}")

        collector.end_run()

        # Override acceptance rate with actual stats from vLLM logs if available
        agg = log_handler.aggregate()
        _detach_log_capture(log_handler)
        if agg and agg.get("acceptance_rate") is not None:
            actual_rate = agg["acceptance_rate"]
            # Backfill per-request acceptance rate proportionally
            for req in collector._requests:
                req.acceptance_rate = actual_rate
            # Mean accepted per step from mean_acceptance_length (= accepted + 1)
            mal = agg.get("mean_acceptance_length", 0)
            # Store aggregated values for reporting
            collector._agg_acceptance_rate = actual_rate
            collector._agg_mean_accepted_per_step = max(0, mal - 1)
            print(f"[Eagle3Runner] Acceptance rate (from logs): {actual_rate:.3f} | "
                  f"Mean acceptance length: {mal:.2f}")

        return collector
