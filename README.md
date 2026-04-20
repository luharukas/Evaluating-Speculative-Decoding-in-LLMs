# Evaluating Speculative Decoding in LLMs

This repository benchmarks standard vLLM decoding against Eagle3 speculative decoding and reports latency, throughput, token acceptance, and output agreement metrics.

The current pipeline is built around:

- A base instruction model served through `vllm.LLM`
- An Eagle3 draft model paired with the same model family
- A config-driven evaluation loop over built-in or custom prompts
- JSON result dumps with both aggregate metrics and optional per-request details

## What It Measures

For each run, the project records:

- Time to first token (TTFT)
- End-to-end generation latency
- Per-request tokens per second
- Overall throughput in tokens per second
- Eagle3 token acceptance rate
- Average accepted draft tokens per step
- Exact-match rate between baseline and speculative outputs

This makes it useful for checking whether speculative decoding gives a real speedup on your hardware and whether greedy outputs stay close to the baseline.

## Repository Layout

```text
.
├── config.yaml          # Default benchmark configuration
├── requirements.txt     # Python dependencies
├── run_pipeline.py      # CLI entrypoint
├── results/             # Saved benchmark JSON files
└── src/
    ├── dataset.py       # Prompt loaders
    ├── metrics.py       # Aggregation and reporting
    ├── pipeline.py      # End-to-end orchestration
    └── runners.py       # Baseline and Eagle3 vLLM runners
```

## How The Pipeline Works

1. Load config from `config.yaml` and apply any CLI overrides.
2. Load prompts from one of three sources:
   - `sharegpt`
   - `mt_bench`
   - `custom`
3. Run a baseline decode pass with the base model.
4. Run an Eagle3 speculative decode pass with the same base model plus a draft model.
5. Aggregate metrics and print a comparison table.
6. Save results to `results/results_<timestamp>.json`.

If only one mode is requested, the pipeline saves a single-run result file instead.

## Requirements

This project assumes a machine capable of running vLLM with the selected models. In practice, that means:

- Linux
- NVIDIA GPU(s) with a CUDA stack compatible with your installed PyTorch and vLLM versions
- Enough VRAM for the base model and, for speculative decoding, the Eagle3 draft model
- Python environment able to install the packages in `requirements.txt`

The default config uses:

- Base model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Draft model: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`

You will typically also need Hugging Face access configured for gated models if your chosen checkpoints require it.

## Installation

Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want the `sharegpt` dataset loader to fetch prompts from Hugging Face, make sure the `datasets` package is installed in your environment as well. If dataset loading fails, the code falls back to the built-in prompt set.

Optional: create a `.env` file for any environment variables needed by your local Hugging Face or vLLM setup. The entrypoint loads `.env` automatically.

## Quick Start

Run the default benchmark:

```bash
python run_pipeline.py
```

Run only baseline decoding:

```bash
python run_pipeline.py --baseline-only
```

Run only Eagle3 speculative decoding:

```bash
python run_pipeline.py --eagle-only
```

Use different checkpoints:

```bash
python run_pipeline.py \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --eagle3-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```

Use fewer prompts for a quick smoke test:

```bash
python run_pipeline.py --num-samples 10 --warmup-runs 1
```

## Configuration

Default settings live in `config.yaml`.

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

output:
  save_dir: "./results"
  save_detailed_json: true
```

Key settings:

- `models.base_model`: main model used for baseline and verification
- `models.eagle3_draft_model`: Eagle3 draft checkpoint, expected to match the base model family
- `speculative_decoding.num_speculative_tokens`: draft tokens proposed per speculative step
- `generation.max_new_tokens`: output length cap per prompt
- `generation.temperature`: defaults to `0.0` for deterministic greedy decoding
- `evaluation.dataset`: `sharegpt`, `mt_bench`, or `custom`
- `evaluation.custom_prompts_path`: JSONL file path for `custom` mode
- `compute.gpu_memory_utilization`: vLLM memory target
- `compute.warmup_runs`: warmup generations before measurement
- `output.save_detailed_json`: whether to store per-request records

## CLI Options

`run_pipeline.py` supports overriding the config from the command line:

```bash
python run_pipeline.py --config my_config.yaml
python run_pipeline.py --num-samples 50
python run_pipeline.py --max-new-tokens 128
python run_pipeline.py --num-speculative-tokens 6
python run_pipeline.py --dataset mt_bench
python run_pipeline.py --prompts data/prompts.jsonl
python run_pipeline.py --gpu-util 0.6
python run_pipeline.py --results-dir ./my_results
python run_pipeline.py --no-save
```

## Datasets

The benchmark supports three prompt sources:

- `sharegpt`: tries to load a ShareGPT-style dataset from Hugging Face
- `mt_bench`: uses a built-in prompt list covering reasoning, coding, writing, and math
- `custom`: reads a JSONL file that contains one prompt per line

Custom prompt file format:

```jsonl
{"prompt": "Explain speculative decoding in simple terms."}
{"prompt": "Write a Python function that reverses a linked list."}
{"text": "Summarize how vLLM computes TTFT."}
```

Both `prompt` and `text` keys are accepted by the loader.

## Output

When both runs are executed, the pipeline writes files like:

```text
results/results_1776654821.json
```

Top-level structure:

```json
{
  "timestamp": 1776654821,
  "baseline": { "... aggregate metrics ..." },
  "eagle3": { "... aggregate metrics ..." }
}
```

Each run includes aggregate fields such as:

- `num_samples`
- `total_time`
- `total_prompt_tokens`
- `total_output_tokens`
- `mean_ttft_ms`, `p50_ttft_ms`, `p90_ttft_ms`, `p99_ttft_ms`
- `mean_e2e_latency_ms`, `p50_e2e_latency_ms`, `p90_e2e_latency_ms`, `p99_e2e_latency_ms`
- `mean_tps`
- `throughput_tps`
- `mean_acceptance_rate` and `mean_accepted_per_step` for Eagle3
- `exact_match_rate` when both baseline and Eagle3 outputs are available

If `save_detailed_json: true`, each run also includes a `per_request` array with:

- prompt text
- generated output text
- prompt and output token counts
- TTFT
- end-to-end latency
- per-request TPS
- speculative decoding estimates such as accepted tokens and acceptance rate

These files can become large because they store full generated text for every prompt.

## Interpreting Results

- Higher `throughput_tps` means better overall generation throughput.
- Lower `mean_e2e_latency_ms` means faster full responses.
- Lower `mean_ttft_ms` means faster first-token responsiveness.
- Higher `mean_acceptance_rate` usually indicates the draft model is helping more.
- Higher `exact_match_rate` means speculative outputs stayed closer to the baseline outputs.

Be careful with `exact_match_rate`: it is a strict string match, not a semantic similarity score.

## Notes And Caveats

- The code is written around the vLLM speculative decoding API used in the current runner implementation.
- `temperature` defaults to `0.0` so output differences mostly reflect decoding-path differences rather than sampling noise.
- The `sharegpt` loader falls back to built-in prompts if Hugging Face dataset loading fails.
- Eagle3 acceptance metrics are estimated per request and then replaced with aggregated log-based values when vLLM emits usable speculative decoding stats.
- Benchmark numbers are hardware-dependent. Compare runs on the same machine and under similar load.

## Typical Workflow

1. Pick a base model and a compatible Eagle3 draft model.
2. Start with `--num-samples 10` to confirm the setup works.
3. Increase to a larger sample count for stable measurements.
4. Compare throughput, latency, and exact-match rate.
5. Tune `num_speculative_tokens` and GPU utilization for your hardware.

## License

No license file is currently included in this repository.
