# Evaluating Speculative Decoding in LLMs

Benchmarks Eagle3 speculative decoding across two inference backends:

- **vLLM** (tested: 0.19.0)
- **TensorRT-LLM** (tested: 1.2.1)

Compares baseline decoding vs Eagle3 speculative decoding, reporting latency, throughput, acceptance rate, and output-agreement metrics.

---

## What It Measures

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token (ms) |
| E2E latency | Full generation latency (ms) |
| TPS | Output tokens per second, per request |
| Throughput | Total tokens per second across all requests |
| Acceptance rate | Fraction of drafted tokens accepted (Eagle3 only) |
| Accepted/step | Average accepted draft tokens per speculative step (Eagle3 only) |
| Exact-match rate | Output agreement rate vs vLLM baseline reference |

---

## Repository Layout

```text
.
├── config.yaml              # Default benchmark configuration
├── requirements-vllm.txt    # vLLM environment snapshot
├── run_pipeline.py          # CLI entrypoint
├── results/                 # Saved benchmark JSON files
└── src/
    ├── dataset.py           # Prompt loaders (ShareGPT, MT-Bench, custom)
    ├── metrics.py           # Aggregation and reporting
    ├── pipeline.py          # End-to-end orchestration
    └── runners.py           # vLLM and TRT-LLM runners
```

---

## How The Pipeline Works

1. Load `config.yaml`, apply any CLI overrides.
2. Load prompts from `sharegpt`, `mt_bench`, or a custom JSONL file.
3. Run any combination of four variants:
   - `vllm_baseline` — vLLM, no speculative decoding
   - `vllm_eagle3` — vLLM + Eagle3 draft model
   - `trtllm_baseline` — TRT-LLM, no speculative decoding
   - `trtllm_eagle3` — TRT-LLM + Eagle3 draft model
4. Aggregate metrics and print a comparison table.
5. Save results to `results/results_<timestamp>.json`.

---

## Prerequisites

- Linux
- NVIDIA GPU with enough VRAM for the base model (and the Eagle3 draft model for speculative runs)
- CUDA stack compatible with the backend you install
- HuggingFace credentials configured if your model checkpoints are gated

Default models used:

| Role | Model |
|------|-------|
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Eagle3 draft | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |

---

## Installation

vLLM and TensorRT-LLM have conflicting dependencies. Use **two separate virtual environments**.

### Shared dependencies (both envs)

```bash
pip install numpy PyYAML python-dotenv tabulate datasets
```

### vLLM environment

```bash
python -m venv .venvvllm
.venvvllm/bin/pip install vllm==0.19.0
.venvvllm/bin/pip install numpy PyYAML python-dotenv tabulate datasets
```

The full vLLM environment snapshot is in `requirements-vllm.txt`.

### TensorRT-LLM environment

```bash
python -m venv .venvtrtllm
.venvtrtllm/bin/pip install tensorrt-llm==1.2.1
.venvtrtllm/bin/pip install numpy PyYAML python-dotenv tabulate datasets ninja
```

> **`ninja` is required.** TensorRT-LLM uses FlashInfer JIT compilation which calls `ninja` at runtime. Install it into the same virtual environment — `pip install ninja`.

### HuggingFace access

Create a `.env` file for tokens if needed:

```bash
HF_TOKEN=hf_...
```

The entrypoint loads `.env` automatically.

---

## Quick Start

Run only TRT-LLM:

```bash
./.venvtrtllm/bin/python run_pipeline.py --backend trtllm
```

Run only vLLM:

```bash
./.venvvllm/bin/python run_pipeline.py --backend vllm
```

Smoke test (fast):

```bash
./.venvtrtllm/bin/python run_pipeline.py --backend trtllm --num-samples 5 --warmup-runs 1
```

Eagle3 only (skip baselines):

```bash
./.venvtrtllm/bin/python run_pipeline.py --backend trtllm --eagle-only
```

---

## Configuration

`config.yaml` controls all defaults:

```yaml
models:
  base_model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  eagle3_draft_model: "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

speculative_decoding:
  num_speculative_tokens: 5
  draft_tensor_parallel_size: 1

generation:
  max_new_tokens: 256
  temperature: 0.0

evaluation:
  dataset: "sharegpt"
  num_samples: 100

compute:
  gpu_memory_utilization: 0.45
  dtype: "float16"
  warmup_runs: 3

tensorrt_llm:
  gpu_memory_utilization: 0.45

output:
  save_dir: "./results"
  save_detailed_json: true
```

Key settings:

| Key | Description |
|-----|-------------|
| `models.base_model` | Main model for baseline and target verification |
| `models.eagle3_draft_model` | Eagle3 draft checkpoint — must match base model family |
| `speculative_decoding.num_speculative_tokens` | Draft tokens proposed per speculative step (5–6 typical) |
| `generation.temperature` | `0.0` = greedy decoding, deterministic output |
| `evaluation.dataset` | `sharegpt`, `mt_bench`, or `custom` |
| `compute.gpu_memory_utilization` | vLLM GPU memory fraction |
| `tensorrt_llm.gpu_memory_utilization` | TRT-LLM GPU memory fraction (overrides compute value if set) |
| `output.save_detailed_json` | Include per-request records in output JSON |

---

## CLI Reference

```bash
# Backend selection
python run_pipeline.py --backend vllm
python run_pipeline.py --backend trtllm
python run_pipeline.py --backend both          # default

# Run selection
python run_pipeline.py --baseline-only         # skip Eagle3 variants
python run_pipeline.py --eagle-only            # skip baseline variants

# Model overrides
python run_pipeline.py --base-model <hf_id_or_path>
python run_pipeline.py --eagle3-model <hf_id_or_path>

# Generation settings
python run_pipeline.py --num-samples 50
python run_pipeline.py --max-new-tokens 128
python run_pipeline.py --num-speculative-tokens 6
python run_pipeline.py --gpu-util 0.6
python run_pipeline.py --warmup-runs 1

# Dataset
python run_pipeline.py --dataset mt_bench
python run_pipeline.py --prompts data/prompts.jsonl

# Output
python run_pipeline.py --results-dir ./my_results
python run_pipeline.py --no-save

# Config file
python run_pipeline.py --config my_config.yaml
```

---

## Datasets

| Source | Description |
|--------|-------------|
| `sharegpt` | ShareGPT-style dataset loaded from HuggingFace; falls back to built-in prompts on failure |
| `mt_bench` | Built-in prompt set covering reasoning, coding, writing, and math |
| `custom` | JSONL file, one prompt per line |

Custom JSONL format (both `prompt` and `text` keys accepted):

```jsonl
{"prompt": "Explain speculative decoding in simple terms."}
{"text": "Write a Python function that reverses a linked list."}
```

---

## Output Format

Results are written to `results/results_<unix_timestamp>.json`:

```json
{
  "timestamp": 1776687816,
  "vllm_baseline": { ... },
  "vllm_eagle3": { ... },
  "trtllm_baseline": { ... },
  "trtllm_eagle3": { ... }
}
```

Each run block contains:

**Aggregate fields:**
`num_samples`, `total_time`, `total_prompt_tokens`, `total_output_tokens`,
`mean_ttft_ms`, `p50_ttft_ms`, `p90_ttft_ms`, `p99_ttft_ms`,
`mean_e2e_latency_ms`, `p50_e2e_latency_ms`, `p90_e2e_latency_ms`, `p99_e2e_latency_ms`,
`mean_tps`, `throughput_tps`,
`mean_acceptance_rate` *(Eagle3 only)*, `mean_accepted_per_step` *(Eagle3 only)*,
`exact_match_rate` *(non-reference runs)*

**Per-request fields** (when `save_detailed_json: true`):
prompt text, generated output, prompt/output token counts, TTFT, E2E latency, TPS, accepted tokens, acceptance rate.

---

## Known Issues and Gotchas

**`ninja` not found at runtime**
TRT-LLM triggers FlashInfer JIT kernel compilation on the first model run. This calls `ninja` as a subprocess. If you invoke the script without activating the venv (`./venv/bin/python run_pipeline.py`), the venv `bin/` is added to PATH automatically — but only if `ninja` was installed into that venv via `pip install ninja`.

**TRT-LLM API version compatibility**
This repo targets TRT-LLM 1.x. The Eagle3 config class is `EagleDecodingConfig` (not `EagleConfig` from older docs). Required fields: `speculative_model`, `max_draft_len`.

**`TLLM_WORKER_USE_SINGLE_PROCESS=1`**
Set automatically for TP=1 runs to avoid MPI worker spawning on single-GPU setups.

**vLLM Eagle3 acceptance stats**
Per-request acceptance rates are estimated heuristically. When vLLM's stat logger emits aggregate spec-decode metrics, those override the estimates.

**TRT-LLM Eagle3 acceptance stats**
Currently heuristic-only per request.

**Benchmark portability**
Numbers are hardware-dependent. Compare runs only on the same machine under similar GPU load.

---

## Typical Workflow

1. Create the venv for your target backend and install dependencies.
2. Smoke-test with `--num-samples 5 --warmup-runs 1`.
3. Run full evaluation with `--num-samples 100` (default).
4. Tune `num_speculative_tokens` (5–6 is typical) and `gpu_memory_utilization` for your GPU.
5. Compare throughput, TTFT, and exact-match rate across backends.

---

## License

No license file is currently included in this repository.
