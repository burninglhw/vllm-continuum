"""Offline prefill profiling for paper Continuum (Appendix C.2).

Run on a reserved idle GPU with the same model/dtype/TP as the serving run.
The measured engine interval is first SCHEDULED -> first generated token;
it excludes queue delay and includes the one-token sampling/engine overhead.
"""
import argparse
# 中文导读：离线标定工具，必须使用 GPU；不是普通 CPU 单元测试。
# 测不同上下文长度的真实 prefill，拟合秒级 a*n^2+b*n+c，供 TTL 策略估价。
import json
import math
import os
from pathlib import Path
import statistics


def context_lengths(max_context):
    result = []
    n = 1000
    while n < max_context:
        result.append(n)
        n *= 2
    # Reserve one token for the first output within the engine's length cap.
    result = sorted(set(result + [max_context - 1]))
    if len(result) < 3:
        raise ValueError("max-context must provide >=3 quadratic fit points")
    return result


def prefill_snapshot(llm):
    metrics = [m for m in llm.get_metrics()
               if m.name == "vllm:request_prefill_time_seconds"]
    return sum(m.count for m in metrics), sum(m.sum for m in metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--kv-gib", type=float, default=4.)
    parser.add_argument("--allocator-gib", type=float, default=26.)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--expected-uuid")
    args = parser.parse_args()
    lengths = context_lengths(args.max_context)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.output.exists():
        parser.error("output already exists; use a new calibration filename")

    import numpy as np
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise ValueError("Allocator cap requires in-process V1 engine")
    device = torch.cuda.get_device_properties(0)
    if args.expected_uuid:
        assert args.expected_uuid.lower().removeprefix("gpu-") in str(device.uuid).lower()
    torch.cuda.set_per_process_memory_fraction(args.allocator_gib * 1024**3 / device.total_memory, 0)

    llm = LLM(model=args.model, dtype=args.dtype,
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_context,
              max_num_batched_tokens=args.max_batched_tokens, max_num_seqs=32,
              kv_cache_memory_bytes=int(args.kv_gib * 1024**3),
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=args.enforce_eager,
              enable_prefix_caching=False, enable_chunked_prefill=True,
              disable_cascade_attn=True, block_size=16,
              scheduling_policy="fcfs", disable_log_stats=False)
    params = SamplingParams(max_tokens=1, temperature=0., ignore_eos=True)
    rows = []
    for n in lengths:
        # 重复 token 只用于测计算形状，不是 SWE 主实验输入，也不用于评价解题能力。
        prompt = {"prompt_token_ids": [10] * n}
        # One warm-up at every size; caching is disabled for every sample.
        llm.generate([prompt], params, use_tqdm=False)
        samples = []
        for _ in range(args.repeats):
            before_count, before_seconds = prefill_snapshot(llm)
            llm.generate([prompt], params, use_tqdm=False)
            count, seconds = prefill_snapshot(llm)
            elapsed = seconds - before_seconds
            if count - before_count != 1 or not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("Missing/invalid V1 prefill metrics; no profile written")
            samples.append(elapsed)
        rows.append({"context_tokens": n, "seconds": samples,
                     "mean_seconds": statistics.mean(samples)})
        print(rows[-1], flush=True)
    x = np.array([row["context_tokens"] for row in rows], dtype=float)
    y = np.array([row["mean_seconds"] for row in rows])
    coefficients = np.polyfit(x, y, 2)
    # 保留原始样本和误差，便于判断共享 GPU 干扰；拟合成功不等于基准实验完成。
    profile = {
        "schema_version": 1, "mode": "prefill", "model": args.model,
        "max_context": args.max_context,
        "coefficients": coefficients.tolist(),
        "fit_rmse_seconds": float(np.sqrt(np.mean((np.polyval(coefficients, x) - y)**2))),
        "gpu_name": torch.cuda.get_device_name(0),
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": str(llm.llm_engine.vllm_config.model_config.dtype),
        "enforce_eager": args.enforce_eager,
        "gpu_uuid": str(device.uuid), "kv_gib": args.kv_gib,
        "max_num_batched_tokens": args.max_batched_tokens,
        "chunked_prefill": True, "max_num_seqs": 32,
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "measurement": "engine_first_scheduled_to_first_token",
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved measured profile: {args.output}")
    llm.llm_engine.engine_core.shutdown()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
