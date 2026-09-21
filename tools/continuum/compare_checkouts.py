"""Run paired, bounded original-vLLM/Continuum trace experiments sequentially.

Both use the same compiled binaries, eager BF16 model, prefix caching, KV
capacity, complete token traces and closed-loop tool durations. Only selected
Python checkout and scheduler differ. Abort the suite if any run/guard fails.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cache-root", required=True, type=Path)
    p.add_argument("--uuid", required=True)
    p.add_argument("--kv-gib", type=float, default=4.)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    for path in (args.data, args.profile, args.upstream / "vllm/__init__.py"):
        if not path.is_file():
            p.error(f"Missing input: {path}")
    args.output.mkdir(exist_ok=False, parents=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=args.uuid, PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_USE_V1="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
               OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               TRITON_CACHE_DIR=str(args.cache_root / "triton"),
               CUDA_CACHE_PATH=str(args.cache_root / "cuda"),
               XDG_CACHE_HOME=str(args.cache_root / "cache"))
    plan = []
    for index, seed in enumerate(args.seeds):
        order = ["upstream", "continuum"] if index % 2 == 0 else ["continuum", "upstream"]
        for implementation in order:
            name = f"{implementation}-seed{seed}"
            selected = args.upstream if implementation == "upstream" else root
            command = [sys.executable, str(root / "tools/continuum/guard_gpu.py"),
                       "--uuid", args.uuid, "--max-own-gib", "28", "--min-free-gib", "100",
                       "--timeout", "720", "--log", str(args.output / (name + ".log")),
                       "--monitor", str(args.output / (name + "-monitor.jsonl")), "--",
                       sys.executable, str(root / "tools/continuum/run_checkout.py"),
                       "--source-root", str(selected), "tools.continuum.verify_gpu",
                       "--model", args.model, "--data", str(args.data),
                       "--profile", str(args.profile), "--output", str(args.output / name),
                       "--implementation", implementation,
                       "--policy", "fcfs" if implementation == "upstream" else "continuum",
                       "--max-context", "16384", "--jps", "4", "--kv-gib", str(args.kv_gib),
                       "--seed", str(seed), "--timeout", "600", "--allocator-gib", "26",
                       "--expected-uuid", args.uuid]
            plan.append({"name": name, "seed": seed, "implementation": implementation, "command": command})
    with (args.output / "plan.json").open("x") as stream:
        json.dump({"kv_gib": args.kv_gib, "order_fixed_before_results": True,
                   "no_future_history_injected": True, "runs": plan}, stream, indent=2)
    results = []
    for run in plan:
        print(f"START {run['name']} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
        code = subprocess.call(run["command"], cwd=root, env=env)
        if code:
            raise SystemExit(f"Stopped suite at {run['name']}: exit={code}; inspect logs/guard")
        summary = json.loads((args.output / run["name"] / "summary.json").read_text())
        assert summary["complete"] and summary["all_tokens_verified"]
        results.append({"name": run["name"], "seed": run["seed"],
                        "implementation": run["implementation"], **summary})
        print(json.dumps(results[-1]), flush=True)
    with (args.output / "comparison.json").open("x") as stream:
        json.dump(results, stream, indent=2)


if __name__ == "__main__":
    main()
