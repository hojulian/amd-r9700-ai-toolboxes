#!/usr/bin/env python3
"""Basic vLLM benchmark runner for AMD R9700 (gfx1201)."""

import argparse
import json
import time
from pathlib import Path

import requests

DEFAULT_HOST = "http://localhost:8000"
PROMPTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the main differences between ROCm and CUDA?",
    "Summarize the history of the AMD GPU architecture.",
]


def get_models(host):
    resp = requests.get(f"{host}/v1/models", timeout=10)
    resp.raise_for_status()
    return [m["id"] for m in resp.json()["data"]]


def run_completion(host, model, prompt, max_tokens=128):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    resp = requests.post(f"{host}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency_s": round(elapsed, 3),
        "tps": round(usage.get("completion_tokens", 0) / elapsed, 1) if elapsed > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM benchmark runner")
    parser.add_argument("--host", default=DEFAULT_HOST, help="vLLM server URL")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--model", default=None, help="Model ID (auto-detected if omitted)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print(f"Connecting to {args.host}...")
    models = get_models(args.host)
    if not models:
        print("No models loaded.")
        return

    model = args.model or models[0]
    print(f"Using model: {model}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {prompt[:60]}...")
        try:
            r = run_completion(args.host, model, prompt, args.max_tokens)
            results.append({"prompt": prompt, **r})
            print(f"  latency={r['latency_s']}s  tps={r['tps']}  tokens={r['completion_tokens']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"prompt": prompt, "error": str(e)})

    valid = [r for r in results if "tps" in r]
    if valid:
        avg_tps = sum(r["tps"] for r in valid) / len(valid)
        avg_lat = sum(r["latency_s"] for r in valid) / len(valid)
        print()
        print(f"Summary: avg_tps={avg_tps:.1f}  avg_latency={avg_lat:.3f}s  runs={len(valid)}/{len(PROMPTS)}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
