"""Benchmark dataset loaders for speculative decoding evaluation."""

import json
import random
from pathlib import Path
from typing import Optional


# Curated MT-Bench style prompts covering reasoning, coding, writing, math
_MT_BENCH_PROMPTS = [
    "Compose a concise poem about the changing seasons, with each stanza representing one season.",
    "Explain the concept of time complexity in algorithms and provide examples for O(1), O(n), and O(n²).",
    "Write a Python function that checks if a given string is a palindrome, handling spaces and punctuation.",
    "What are the key differences between supervised and unsupervised machine learning? Give concrete examples.",
    "Explain how the TCP/IP protocol suite works, from the application layer down to the physical layer.",
    "Write a short story about an astronaut who discovers an unexpected life form on Mars.",
    "Solve this problem: A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What is the average speed?",
    "What are the ethical implications of using AI in criminal justice systems? Present multiple perspectives.",
    "Write a recursive Python function to compute the nth Fibonacci number with memoization.",
    "Explain the difference between SQL and NoSQL databases. When would you choose each?",
    "Describe the water cycle and explain why it is critical for life on Earth.",
    "What is the significance of the Turing Test and what are its limitations as a measure of intelligence?",
    "Write a Python class implementing a stack data structure with push, pop, and peek operations.",
    "Explain quantum entanglement in simple terms, as if teaching a curious 12-year-old.",
    "What were the main causes of the First World War? Analyze the role of nationalism and alliances.",
    "Write a function to find all prime numbers up to n using the Sieve of Eratosthenes.",
    "How does HTTPS ensure secure communication? Explain TLS handshake briefly.",
    "Compare object-oriented and functional programming paradigms with practical code examples.",
    "What is the difference between correlation and causation? Give a real-world example of each.",
    "Explain how transformer architecture changed natural language processing. What problem did attention solve?",
]

_CODING_PROMPTS = [
    "Write a Python function to merge two sorted arrays into a single sorted array without using sort().",
    "Implement a binary search tree in Python with insert, search, and in-order traversal methods.",
    "Write a Python decorator that caches function results (memoization) using a dictionary.",
    "Implement a LRU cache in Python with O(1) get and put operations.",
    "Write a Python function that finds the longest common subsequence of two strings.",
    "Implement quicksort in Python. Explain your pivot selection strategy.",
    "Write a Python context manager that measures and prints function execution time.",
    "Implement a simple tokenizer in Python that splits text into sentences and words.",
    "Write a Python function to validate an email address using only string operations (no regex).",
    "Implement a graph class in Python and add BFS and DFS traversal methods.",
]

_REASONING_PROMPTS = [
    "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops definitely Lazzles? Explain.",
    "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "There are 3 boxes: one has only apples, one only oranges, one has both. All labels are wrong. You can pick one fruit from one box. How do you label them correctly?",
    "You have 8 balls, one is slightly heavier. Using a balance scale only twice, find the heavy ball.",
    "In a town, the barber shaves all those who do not shave themselves. Does the barber shave himself?",
]


def load_sharegpt(num_samples: int, seed: int = 42) -> list[str]:
    """Load ShareGPT-style prompts from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                          data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                          split="train")
        random.seed(seed)
        indices = random.sample(range(len(ds)), min(num_samples * 3, len(ds)))
        prompts = []
        for i in indices:
            convs = ds[i].get("conversations", [])
            if convs and convs[0].get("from") == "human":
                text = convs[0].get("value", "").strip()
                if 20 < len(text) < 1000:
                    prompts.append(text)
            if len(prompts) >= num_samples:
                break
        return prompts[:num_samples]
    except Exception:
        return load_builtin(num_samples, seed)


def load_builtin(num_samples: int, seed: int = 42) -> list[str]:
    """Fall back to built-in benchmark prompts."""
    all_prompts = _MT_BENCH_PROMPTS + _CODING_PROMPTS + _REASONING_PROMPTS
    random.seed(seed)
    if num_samples <= len(all_prompts):
        return random.sample(all_prompts, num_samples)
    # Repeat with shuffling if more samples requested
    result = []
    while len(result) < num_samples:
        shuffled = all_prompts[:]
        random.shuffle(shuffled)
        result.extend(shuffled)
    return result[:num_samples]


def load_mt_bench(num_samples: int, seed: int = 42) -> list[str]:
    """Load MT-Bench prompts (built-in subset)."""
    return load_builtin(num_samples, seed)


def load_custom(path: str, num_samples: int) -> list[str]:
    """Load prompts from a JSONL file with {"prompt": "..."} per line."""
    prompts = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if "prompt" in obj:
                prompts.append(obj["prompt"])
            elif "text" in obj:
                prompts.append(obj["text"])
            if len(prompts) >= num_samples:
                break
    return prompts


def load_dataset_prompts(
    dataset: str,
    num_samples: int,
    custom_path: Optional[str] = None,
    seed: int = 42,
) -> list[str]:
    """Dispatch to the correct dataset loader."""
    if dataset == "sharegpt":
        prompts = load_sharegpt(num_samples, seed)
    elif dataset == "mt_bench":
        prompts = load_mt_bench(num_samples, seed)
    elif dataset == "custom":
        if not custom_path:
            raise ValueError("custom_prompts_path required for dataset='custom'")
        prompts = load_custom(custom_path, num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: sharegpt, mt_bench, custom")

    if len(prompts) < num_samples:
        print(f"[dataset] Warning: only {len(prompts)} prompts available (requested {num_samples})")
    return prompts
