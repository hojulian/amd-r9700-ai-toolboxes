#!/usr/bin/env python3
"""Interactive TUI launcher for vLLM on AMD R9700 (gfx1201)."""

import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS_FILE = Path("/opt/max_context_results.json")
DEFAULT_GPU_UTIL = "0.95"
DEFAULT_DTYPE = "auto"


def run_dialog(args):
    try:
        result = subprocess.run(
            ["dialog"] + args,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode, result.stderr.strip()
    except FileNotFoundError:
        print("ERROR: 'dialog' not found. Install it with: dnf install dialog")
        sys.exit(1)


def detect_gpus():
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showid", "--csv"], text=True, stderr=subprocess.DEVNULL
        )
        count = len([l for l in out.strip().splitlines() if l and not l.startswith("device")])
        return max(count, 1)
    except Exception:
        devs = list(Path("/dev/dri").glob("renderD*"))
        return max(len(devs), 1)


def load_results():
    if not RESULTS_FILE.exists():
        return []
    try:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_verified_config(results, model_name, tp, max_reqs):
    for entry in results:
        if (
            entry.get("model", "").lower() in model_name.lower()
            and entry.get("tensor_parallel_size") == tp
            and entry.get("max_num_seqs") == max_reqs
        ):
            return entry
    return None


def msgbox(title, text):
    run_dialog(["--title", title, "--msgbox", text, "10", "60"])


def inputbox(title, prompt, default=""):
    code, val = run_dialog(["--title", title, "--inputbox", prompt, "8", "60", default])
    return val if code == 0 else None


def menu(title, prompt, items):
    flat = []
    for tag, desc in items:
        flat += [tag, desc]
    code, val = run_dialog(["--title", title, "--menu", prompt, "20", "70", "10"] + flat)
    return val if code == 0 else None


def yesno(title, question):
    code, _ = run_dialog(["--title", title, "--yesno", question, "7", "60"])
    return code == 0


def nuke_vllm_cache():
    cache_dir = Path.home() / ".cache" / "vllm"
    if cache_dir.exists():
        subprocess.run(["rm", "-rf", str(cache_dir)], check=False)
        msgbox("Cache Cleared", f"vLLM cache removed:\n{cache_dir}")
    else:
        msgbox("Cache Clear", "No vLLM cache found.")


def main():
    num_gpus = detect_gpus()
    results = load_results()

    model_path = inputbox("Model Path", "Enter path or HuggingFace model ID:", "")
    if not model_path:
        sys.exit(0)

    tp_options = [(str(i), f"{i} GPU(s)") for i in range(1, num_gpus + 1)]
    tp_str = menu("Tensor Parallelism", "Select number of GPUs:", tp_options)
    if not tp_str:
        sys.exit(0)
    tp = int(tp_str)

    concurrency_options = [("1", "1 concurrent request"), ("4", "4"), ("8", "8"), ("16", "16"), ("32", "32")]
    max_reqs_str = menu("Concurrency", "Max concurrent requests:", concurrency_options)
    if not max_reqs_str:
        sys.exit(0)
    max_reqs = int(max_reqs_str)

    verified = get_verified_config(results, model_path, tp, max_reqs)
    ctx_default = str(verified.get("max_model_len", 8192)) if verified else "8192"
    ctx_input = inputbox("Context Length", "Max context length (tokens):", ctx_default)
    if not ctx_input:
        sys.exit(0)
    max_model_len = int(ctx_input)

    gpu_util_input = inputbox("GPU Memory Utilization", "GPU memory utilization (0.0-1.0):", DEFAULT_GPU_UTIL)
    if not gpu_util_input:
        sys.exit(0)

    attn_options = [
        ("PAGED_ATTENTION", "ROCm paged attention (stable)"),
        ("TRITON", "Triton flash attention (faster, experimental)"),
    ]
    attn = menu("Attention Backend", "Select attention backend:", attn_options)
    if not attn:
        sys.exit(0)

    if yesno("Clear Cache", "Clear vLLM compile cache before starting?"):
        nuke_vllm_cache()

    env = os.environ.copy()
    env["VLLM_TARGET_DEVICE"] = "rocm"
    env["PYTORCH_ROCM_ARCH"] = "gfx1201"
    if attn == "TRITON":
        env["VLLM_USE_TRITON_FLASH_ATTN"] = "1"
        env["VLLM_V1_USE_PREFILL_DECODE_ATTENTION"] = "1"
    else:
        env.pop("VLLM_USE_TRITON_FLASH_ATTN", None)

    cmd = [
        "vllm", "serve", model_path,
        "--tensor-parallel-size", str(tp),
        "--max-num-seqs", str(max_reqs),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", gpu_util_input,
        "--dtype", DEFAULT_DTYPE,
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    print("\nLaunching vLLM:")
    print("  " + " ".join(cmd))
    print()
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
