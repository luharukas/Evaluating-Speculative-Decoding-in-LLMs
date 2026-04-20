"""Metrics collection and reporting for speculative decoding evaluation."""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


@dataclass
class RequestMetrics:
    """Per-request timing and token metrics."""
    prompt: str
    output_text: str
    prompt_tokens: int
    output_tokens: int
    # Timings in seconds
    ttft: float              # Time to first token
    e2e_latency: float       # Total generation time
    tps: float               # Tokens per second for this request
    # Speculative decoding specific (filled only for Eagle3 run)
    num_draft_steps: Optional[int] = None
    accepted_tokens: Optional[int] = None
    acceptance_rate: Optional[float] = None


@dataclass
class RunMetrics:
    """Aggregated metrics for a full run (baseline or Eagle3)."""
    run_type: str            # "baseline" or "eagle3"
    model: str
    draft_model: Optional[str]
    num_samples: int
    total_time: float        # Wall-clock time for all generations

    # Token counts
    total_prompt_tokens: int
    total_output_tokens: int

    # Speed metrics (ms / tok/s)
    mean_ttft_ms: float
    p50_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float

    mean_e2e_latency_ms: float
    p50_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    p99_e2e_latency_ms: float

    mean_tps: float           # Per-request tokens/sec
    throughput_tps: float     # Total tokens / total wall time

    # Eagle3-specific
    mean_acceptance_rate: Optional[float] = None
    mean_accepted_per_step: Optional[float] = None

    # Quality (vs baseline)
    exact_match_rate: Optional[float] = None  # Filled when comparing

    per_request: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("per_request", None)
        return d


class MetricsCollector:
    """Collect metrics during a run and build RunMetrics."""

    def __init__(self, run_type: str, model: str, draft_model: Optional[str] = None):
        self.run_type = run_type
        self.model = model
        self.draft_model = draft_model
        self._requests: list[RequestMetrics] = []
        self._run_start: Optional[float] = None
        self._run_end: Optional[float] = None
        # Filled by Eagle3Runner when accurate log-based stats are available
        self._agg_acceptance_rate: Optional[float] = None
        self._agg_mean_accepted_per_step: Optional[float] = None

    def start_run(self):
        self._run_start = time.perf_counter()

    def end_run(self):
        self._run_end = time.perf_counter()

    def add_request(self, req: RequestMetrics):
        self._requests.append(req)

    def build(self) -> RunMetrics:
        assert self._run_start and self._run_end, "Call start_run/end_run first"
        reqs = self._requests
        n = len(reqs)

        ttfts = np.array([r.ttft * 1000 for r in reqs])        # → ms
        latencies = np.array([r.e2e_latency * 1000 for r in reqs])  # → ms
        tps_list = np.array([r.tps for r in reqs])

        total_output_tokens = sum(r.output_tokens for r in reqs)
        total_wall = self._run_end - self._run_start

        # Acceptance rate (Eagle3 only)
        acc_rates = [r.acceptance_rate for r in reqs if r.acceptance_rate is not None]
        accepted_per_step = []
        for r in reqs:
            if r.num_draft_steps and r.num_draft_steps > 0 and r.accepted_tokens is not None:
                accepted_per_step.append(r.accepted_tokens / r.num_draft_steps)

        return RunMetrics(
            run_type=self.run_type,
            model=self.model,
            draft_model=self.draft_model,
            num_samples=n,
            total_time=total_wall,
            total_prompt_tokens=sum(r.prompt_tokens for r in reqs),
            total_output_tokens=total_output_tokens,
            mean_ttft_ms=float(np.mean(ttfts)),
            p50_ttft_ms=float(np.percentile(ttfts, 50)),
            p90_ttft_ms=float(np.percentile(ttfts, 90)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            mean_e2e_latency_ms=float(np.mean(latencies)),
            p50_e2e_latency_ms=float(np.percentile(latencies, 50)),
            p90_e2e_latency_ms=float(np.percentile(latencies, 90)),
            p99_e2e_latency_ms=float(np.percentile(latencies, 99)),
            mean_tps=float(np.mean(tps_list)),
            throughput_tps=total_output_tokens / total_wall if total_wall > 0 else 0,
            # Prefer log-captured accurate stats over per-request estimates
            mean_acceptance_rate=(
                self._agg_acceptance_rate
                if self._agg_acceptance_rate is not None
                else (float(np.mean(acc_rates)) if acc_rates else None)
            ),
            mean_accepted_per_step=(
                self._agg_mean_accepted_per_step
                if self._agg_mean_accepted_per_step is not None
                else (float(np.mean(accepted_per_step)) if accepted_per_step else None)
            ),
            per_request=[vars(r) for r in reqs],
        )


def compare_outputs(baseline: RunMetrics, eagle: RunMetrics) -> float:
    """Compute exact-match rate between baseline and Eagle3 outputs."""
    b_outputs = [r["output_text"].strip() for r in baseline.per_request]
    e_outputs = [r["output_text"].strip() for r in eagle.per_request]
    if not b_outputs or not e_outputs:
        return 0.0
    matches = sum(b == e for b, e in zip(b_outputs, e_outputs))
    return matches / len(b_outputs)


_RUN_LABELS = {
    "baseline": "vLLM Baseline",
    "eagle3": "vLLM Eagle3",
    "trtllm_baseline": "TRT-LLM Baseline",
    "trtllm_eagle3": "TRT-LLM Eagle3",
}


def print_multi_run_table(runs: list):
    """Print N-way comparison table. Speedup columns computed vs first run."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    if not runs:
        return

    headers = ["Metric"] + [_RUN_LABELS.get(r.run_type, r.run_type) for r in runs]

    def _speedup(values: list, higher_is_better: bool) -> list:
        base = values[0]
        out: list = []
        for i, v in enumerate(values):
            if i == 0 or v is None or not base:
                out.append("—")
            else:
                ratio = (v / base) if higher_is_better else (base / v)
                out.append(f"{ratio:.2f}x")
        return out

    rows = []

    # TTFT
    rows.append(["Mean TTFT (ms)"] + [f"{r.mean_ttft_ms:.1f}" for r in runs])
    rows.append(["P50 TTFT (ms)"]  + [f"{r.p50_ttft_ms:.1f}"  for r in runs])
    rows.append(["P90 TTFT (ms)"]  + [f"{r.p90_ttft_ms:.1f}"  for r in runs])
    rows.append(["P99 TTFT (ms)"]  + [f"{r.p99_ttft_ms:.1f}"  for r in runs])

    # E2E latency
    rows.append(["Mean E2E Latency (ms)"] + [f"{r.mean_e2e_latency_ms:.1f}" for r in runs])
    rows.append(["P50 E2E Latency (ms)"]  + [f"{r.p50_e2e_latency_ms:.1f}" for r in runs])
    rows.append(["P90 E2E Latency (ms)"]  + [f"{r.p90_e2e_latency_ms:.1f}" for r in runs])
    rows.append(
        ["  E2E Speedup vs [0]"]
        + _speedup([r.mean_e2e_latency_ms for r in runs], higher_is_better=False)
    )

    # Throughput
    rows.append(["Mean TPS (per-req)"] + [f"{r.mean_tps:.1f}" for r in runs])
    rows.append(["Throughput (tok/s)"] + [f"{r.throughput_tps:.1f}" for r in runs])
    rows.append(
        ["  Throughput Speedup vs [0]"]
        + _speedup([r.throughput_tps for r in runs], higher_is_better=True)
    )

    # Counts & time
    rows.append(["Total Output Tokens"] + [str(r.total_output_tokens) for r in runs])
    rows.append(["Total Time (s)"]      + [f"{r.total_time:.1f}" for r in runs])

    # Spec-decoding stats
    rows.append([
        "Token Acceptance Rate"
    ] + [
        f"{r.mean_acceptance_rate:.3f}" if r.mean_acceptance_rate is not None else "N/A"
        for r in runs
    ])
    rows.append([
        "Avg Accepted Tok/Step"
    ] + [
        f"{r.mean_accepted_per_step:.2f}" if r.mean_accepted_per_step is not None else "N/A"
        for r in runs
    ])

    # Exact match vs first run
    rows.append([
        "Output Exact Match vs [0]"
    ] + [
        "—" if i == 0
        else (f"{r.exact_match_rate:.1%}" if r.exact_match_rate is not None else "N/A")
        for i, r in enumerate(runs)
    ])

    sep = "=" * (28 + 16 * len(runs))
    print(f"\n{sep}")
    print("  INFERENCE BACKEND COMPARISON")
    print(sep)
    print(f"  Base model : {runs[0].model}")
    draft_models = [r.draft_model for r in runs if r.draft_model]
    if draft_models:
        print(f"  Draft model: {draft_models[0]}")
    print(f"  Samples    : {runs[0].num_samples}")
    print(sep)
    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [28] + [16] * len(runs)
        print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
        print("  " + "-" * (sum(col_w) + len(col_w) * 2))
        for row in rows:
            print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))
    print(sep + "\n")


def print_comparison_table(baseline: RunMetrics, eagle: RunMetrics):
    """Print a formatted side-by-side comparison table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    speedup_tps = eagle.throughput_tps / baseline.throughput_tps if baseline.throughput_tps else 0
    speedup_latency = baseline.mean_e2e_latency_ms / eagle.mean_e2e_latency_ms if eagle.mean_e2e_latency_ms else 0

    rows = [
        ["Metric", "Baseline", "Eagle3", "Speedup / Delta"],
        ["---", "---", "---", "---"],
        ["Mean TTFT (ms)",
         f"{baseline.mean_ttft_ms:.1f}",
         f"{eagle.mean_ttft_ms:.1f}",
         f"{baseline.mean_ttft_ms / eagle.mean_ttft_ms:.2f}x faster" if eagle.mean_ttft_ms else "N/A"],
        ["P50 TTFT (ms)", f"{baseline.p50_ttft_ms:.1f}", f"{eagle.p50_ttft_ms:.1f}", ""],
        ["P90 TTFT (ms)", f"{baseline.p90_ttft_ms:.1f}", f"{eagle.p90_ttft_ms:.1f}", ""],
        ["P99 TTFT (ms)", f"{baseline.p99_ttft_ms:.1f}", f"{eagle.p99_ttft_ms:.1f}", ""],
        ["Mean E2E Latency (ms)",
         f"{baseline.mean_e2e_latency_ms:.1f}",
         f"{eagle.mean_e2e_latency_ms:.1f}",
         f"{speedup_latency:.2f}x faster"],
        ["P50 E2E Latency (ms)", f"{baseline.p50_e2e_latency_ms:.1f}", f"{eagle.p50_e2e_latency_ms:.1f}", ""],
        ["P90 E2E Latency (ms)", f"{baseline.p90_e2e_latency_ms:.1f}", f"{eagle.p90_e2e_latency_ms:.1f}", ""],
        ["Mean TPS (per request)",
         f"{baseline.mean_tps:.1f}",
         f"{eagle.mean_tps:.1f}",
         f"{eagle.mean_tps / baseline.mean_tps:.2f}x"],
        ["Throughput (tok/s)",
         f"{baseline.throughput_tps:.1f}",
         f"{eagle.throughput_tps:.1f}",
         f"{speedup_tps:.2f}x"],
        ["Total Output Tokens",
         str(baseline.total_output_tokens),
         str(eagle.total_output_tokens),
         ""],
        ["Total Time (s)",
         f"{baseline.total_time:.1f}",
         f"{eagle.total_time:.1f}",
         ""],
        ["Token Acceptance Rate",
         "N/A",
         f"{eagle.mean_acceptance_rate:.3f}" if eagle.mean_acceptance_rate is not None else "N/A",
         ""],
        ["Avg Accepted Tok/Step",
         "N/A",
         f"{eagle.mean_accepted_per_step:.2f}" if eagle.mean_accepted_per_step is not None else "N/A",
         ""],
        ["Output Exact Match",
         "—",
         f"{eagle.exact_match_rate:.1%}" if eagle.exact_match_rate is not None else "N/A",
         "Correctness check"],
    ]

    print("\n" + "=" * 70)
    print("  SPECULATIVE DECODING EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Base model  : {baseline.model}")
    print(f"  Draft model : {eagle.draft_model or 'N/A'}")
    print(f"  Samples     : {baseline.num_samples}")
    print("=" * 70)

    if tabulate:
        print(tabulate(rows[2:], headers=rows[0], tablefmt="rounded_outline"))
    else:
        col_w = [25, 12, 12, 22]
        header = rows[0]
        print("  " + "  ".join(h.ljust(w) for h, w in zip(header, col_w)))
        print("  " + "-" * (sum(col_w) + len(col_w) * 2))
        for row in rows[2:]:
            print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))
    print("=" * 70 + "\n")
