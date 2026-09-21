"""Bounded real-forward SWE trace replay using the ORIGINAL vLLM Scheduler.

Not an end-to-end task quality benchmark or a reproduction of paper figures.
Complete selected programs run by default; explicit max-turns may select
prefixes. Context overflow is an error, never silent truncation. Final
turns are marked terminal AFTER completion; total lengths/tool times are not
passed to the scheduling policy. Pickle globals are forbidden when reading the
existing locally prepared token dataset, whose manifest hash is verified.
"""
import argparse
import hashlib
import heapq
import inspect
import json
import os
import random
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

from tools.continuum.trace_dataset import load_replay

# 中文导读：这是实验驱动器，不是新调度算法。它把固定输入送入同一个真实
# vLLM 模型，按 seed 生成初始到达间隔，按记录的工具时长安排后续返回。
# 改变 --policy 只切换 FCFS/Continuum；输出被约束以保证两边后续前缀一致。


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (index - lower) * (values[upper] - values[lower])


def load_workload(args):
    original, manifest = load_replay(args.data)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert digest(Path(args.model) / name) == manifest["tokenizer_file_hashes"][name], name
    programs = []
    for program in original[:args.programs]:
        steps = []
        for step in program["steps"][:args.max_turns]:
            if len(step["prompt"]) + len(step["output"]) + 1 > args.max_context:
                raise ValueError(f"Context overflow in complete trace {program['instance_id']}; "
                                 "choose a fitting dataset or a calibrated larger context")
            assert step["output"] and step["gap"] >= 0
            steps.append(step)
        if not steps:
            raise ValueError(f"First request outside context: {program['instance_id']}")
        programs.append({"instance_id": program["instance_id"], "steps": steps,
                         "original_turns": len(program["steps"])})
    return programs, manifest


def scheduled_prefill_tokens(prompt_tokens, computed_after, scheduled):
    # schedule() 返回前已经把 num_computed_tokens 加上本 tick 的数量，
    # 所以统计本轮实际 prefill 必须回到 begin=after-scheduled，不能从 after 再算。
    begin = computed_after - scheduled
    return max(0, min(computed_after, prompt_tokens) - min(begin, prompt_tokens))


def run(args):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise ValueError("Memory cap/instrumentation require in-process V1 engine")
    if args.expected_uuid:
        assert args.expected_uuid.lower().removeprefix("gpu-") in str(torch.cuda.get_device_properties(0).uuid).lower()
    if args.allocator_gib:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.allocator_gib * 1024**3 / total, 0)

    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["RUN_OUTPUT_DIR"] = str(args.output.resolve())
    programs, manifest = load_workload(args)
    # 这是“到达随机种子”，不是抽数据的种子，也不是模型采样种子。
    rng = random.Random(args.seed)
    arrivals, offset = [], 0.
    for i in range(len(programs)):
        if i:
            offset += rng.expovariate(args.jps)
        arrivals.append(offset)
    helper_root = Path(__file__).resolve().parents[2]
    source_root = Path(os.environ.get("BENCHMARK_SOURCE_ROOT", str(helper_root))).resolve()
    source_paths = [
        "vllm/v1/core/sched/scheduler.py", "vllm/v1/core/sched/request_queue.py",
        "vllm/v1/core/kv_cache_manager.py", "vllm/v1/engine/core.py", "vllm/v1/request.py"]
    if args.implementation == "continuum":
        source_paths += ["vllm/v1/core/estimate_with_func.py", "vllm/v1/core/continuum_policy.py"]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        source_sha256={p: digest(source_root / p) for p in source_paths},
        harness_sha256={p: digest(helper_root / "tools/continuum" / p)
                        for p in ("verify_gpu.py", "replay_tokens.py", "trace_dataset.py")},
        profile_sha256=digest(args.profile), replay_sha256=manifest["replay_sha256"],
        tokenizer_verified=True, arrival_offsets=arrivals,
        programs_selected=[{"id": p["instance_id"], "turns": len(p["steps"]),
                            "original_turns": p["original_turns"]} for p in programs],
        limitation="Fixed traces, forced decode after real forward; no quality claim",
        whole_programs_preserved=all(len(p["steps"]) == p["original_turns"] for p in programs),
        tool_timing="Original recorded durations, including final selected response's tool gap",
        scheduler_source=inspect.getfile(Scheduler))
    save(args.output / "config.json", config)
    model = None
    try:
        model = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=1,
            max_model_len=args.max_context, max_num_batched_tokens=2048,
            max_num_seqs=32, enable_prefix_caching=True, block_size=16,
            enable_chunked_prefill=True, kv_cache_memory_bytes=int(args.kv_gib * 1024**3),
            gpu_memory_utilization=.20, enforce_eager=True,
            disable_cascade_attn=True, disable_log_stats=False,
            scheduling_policy=args.policy,
            additional_config={"continuum": {"profile_path": str(args.profile)}} if args.policy == "continuum" else {},
            logits_processors=["tools.continuum.replay_tokens:ReplayTokens"])
        engine = model.llm_engine
        scheduler = engine.engine_core.engine_core.scheduler
        assert type(scheduler) is Scheduler, type(scheduler)
        assert Path(inspect.getfile(type(scheduler))).resolve().is_relative_to(source_root)
        estimator = getattr(scheduler, "tool_call_estimator", None)
        assert (estimator is not None) == (args.policy == "continuum")
        if args.implementation == "upstream":
            assert not hasattr(scheduler, "tool_call_estimator"), "Not pristine upstream scheduler"
        device = torch.cuda.get_device_properties(0)
        if args.expected_uuid:
            assert args.expected_uuid.lower().removeprefix("gpu-") in str(device.uuid).lower()
        save(args.output / "device.json", {"properties": str(device), "uuid": str(device.uuid),
                                          "num_gpu_blocks": scheduler.kv_cache_manager.block_pool.num_gpu_blocks})
        eos_ids = engine.vllm_config.model_config.hf_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
        for p in programs:
            for step in p["steps"]:
                assert step["output"][-1] in eos_ids, "Trace must end with model EOS"
                assert not set(step["output"][:-1]).intersection(eos_ids), "Early EOS in trace"
        warm = SamplingParams(temperature=0, max_tokens=17, ignore_eos=True,
                              detokenize=False)
        model.generate([{"prompt_token_ids": [10] * 2048}], warm, use_tqdm=False)
        engine.reset_prefix_cache()
        recorder = getattr(scheduler, "continuum_recorder", None)
        if recorder is not None:
            recorder.job_id_to_history.clear()
            scheduler.running_job_id_first_entry_time.clear()
        assert not getattr(scheduler, "pinned_requests", {}) and not scheduler.requests
        if estimator:
            assert not estimator.programs
            policy = estimator.policy
            policy.completed_lengths.clear()
            policy.queue_delays.clear()
            policy.tool_durations.clear()
        schedule_times = []
        counts = {"executed_prefill_tokens": 0, "scheduled_tokens": 0,
                  "max_active_blocks": 0, "deadlock_unpins": 0, "preemptions": 0}
        real_schedule = scheduler.schedule
        real_deadlock = getattr(scheduler, "_unpin_latest_program", None)

        def measured_schedule():
            running_before = {r.request_id for r in scheduler.running}
            before = time.perf_counter()
            result = real_schedule()
            schedule_times.append(time.perf_counter() - before)
            counts["preemptions"] += sum(
                r.request_id in running_before and r.status == RequestStatus.PREEMPTED
                for r in scheduler.waiting)
            for request_id, count in result.num_scheduled_tokens.items():
                req = scheduler.requests[request_id]
                counts["scheduled_tokens"] += count
                counts["executed_prefill_tokens"] += scheduled_prefill_tokens(
                    req.num_prompt_tokens, req.num_computed_tokens, count)
            pool = scheduler.kv_cache_manager.block_pool
            counts["max_active_blocks"] = max(counts["max_active_blocks"],
                pool.num_gpu_blocks - 1 - pool.get_num_free_blocks())
            return result

        def measured_deadlock():
            released = real_deadlock()
            counts["deadlock_unpins"] += int(released)
            return released

        scheduler.schedule = measured_schedule
        if real_deadlock is not None:
            scheduler._unpin_latest_program = measured_deadlock
        start = time.time()
        pending = [(start + a, i, 0) for i, a in enumerate(arrivals)]
        heapq.heapify(pending)
        active, turns, jobs = {}, [], []
        with (args.output / "turns.jsonl").open("x") as turn_log:
            while pending or active:
                if time.time() - start > args.timeout:
                    raise TimeoutError("Replay time budget exceeded")
                while pending and pending[0][0] <= time.time():
                    due, pi, si = heapq.heappop(pending)
                    program = programs[pi]
                    if si == len(program["steps"]):
                        # 最后一次工具调用也属于任务；与旧完整回放的 JCT 口径一致。
                        job = {"id": program["instance_id"], "jct": due - start - arrivals[pi]}
                        jobs.append(job)
                        print(json.dumps({"completed": len(jobs), "total": len(programs), **job}), flush=True)
                        continue
                    step = program["steps"][si]
                    request_id = f"p{pi}-s{si}"
                    params = SamplingParams(
                        temperature=0, max_tokens=len(step["output"]) + 1,
                        ignore_eos=False, detokenize=False,
                        output_kind=RequestOutputKind.FINAL_ONLY,
                        extra_args={"job_id": program["instance_id"],
                                    "is_last_step": si == len(program["steps"]) - 1,
                                    "replay_token_ids": step["output"]})
                    submitted = time.time()
                    engine.add_request(request_id, {"prompt_token_ids": step["prompt"]},
                                       params, arrival_time=submitted)
                    active[request_id] = (pi, si, due, submitted)
                if active:
                    for result in engine.step():
                        if not result.finished:
                            continue
                        finished = time.time()
                        pi, si, due, submitted = active.pop(result.request_id)
                        program = programs[pi]
                        step = program["steps"][si]
                        completion = result.outputs[0]
                        assert list(completion.token_ids) == step["output"], result.request_id
                        assert completion.finish_reason == "stop", completion.finish_reason
                        row = {"id": program["instance_id"], "step": si,
                               "request_id": result.request_id, "due": due - start,
                               "submitted": submitted - start, "finished": finished - start,
                               "request_seconds": finished - submitted,
                               "dispatch_lag": submitted - due,
                               "prompt_tokens": len(step["prompt"]),
                               "output_tokens": len(step["output"]),
                               "hit_tokens": result.num_cached_tokens or 0,
                               "verified": True, "finish_reason": completion.finish_reason}
                        turns.append(row)
                        turn_log.write(json.dumps(row) + "\n")
                        turn_log.flush()
                        heapq.heappush(pending, (finished + step["gap"], pi, si + 1))
                elif pending:
                    if hasattr(scheduler, "unpin_requests_regular"):
                        scheduler.unpin_requests_regular()
                    time.sleep(max(0, min(.01, pending[0][0] - time.time())))
        elapsed = time.time() - start
        assert not scheduler.requests and not scheduler.waiting and not scheduler.running
        assert not getattr(scheduler, "pinned_requests", {}), "Unreleased pins after all terminal responses"
        pool = scheduler.kv_cache_manager.block_pool
        assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null), "KV reference leak"
        if estimator:
            assert not estimator.programs
        history = recorder.job_id_to_history if recorder is not None else {}
        events = [e for values in history.values() for e in values]
        ttls = [e["ttl_decision"] for e in events if "ttl_decision" in e]
        admission = [e for e in events if "waiting_to_running" in e or "evicted_to_running" in e]
        summary = {
            "complete": True, "programs": len(jobs), "responses": len(turns),
            "all_tokens_verified": True, "all_finished_by_eos": True,
            "no_pins_or_kv_reference_leaks": True,
            "mean_jct": statistics.mean(j["jct"] for j in jobs),
            "p50_jct": percentile([j["jct"] for j in jobs], .5),
            "p95_jct": percentile([j["jct"] for j in jobs], .95),
            "makespan": elapsed, "jobs_per_second": len(jobs) / elapsed,
            "mean_request_seconds": statistics.mean(r["request_seconds"] for r in turns),
            "prompt_tokens": sum(r["prompt_tokens"] for r in turns),
            "output_tokens": sum(r["output_tokens"] for r in turns),
            "request_cached_tokens": sum(r["hit_tokens"] for r in turns),
            "admission_hit_tokens": sum(e["hit_length"] for e in admission) if recorder else None,
            "recorder_preemptions": sum("Request_evicted_from_running_queue_time" in e for e in events) if recorder else None,
            "pin_events": sum("pinned_time" in e for e in events),
            "unpin_events": sum("unpinned_time" in e for e in events),
            "ttl_decisions": len(ttls), "positive_ttls": sum(t["ttl"] > 0 for t in ttls),
            "ttl_estimation_sources": dict(Counter(t["estimation_source"] for t in ttls)),
            "positive_ttl_sources": dict(Counter(t["estimation_source"] for t in ttls if t["ttl"] > 0)),
            "tool_return_samples": sum(len(v) for v in estimator.policy.tool_durations.values()) if estimator else None,
            # history 按程序分组，不是全局时间序；不能用 ttls[-1] 冒充最后决策。
            "eta_final_empirical": estimator.policy.eta if estimator else None,
            "mean_scheduler_ms": statistics.mean(schedule_times) * 1000,
            "p95_scheduler_ms": percentile(schedule_times, .95) * 1000,
            "max_dispatch_lag": max(r["dispatch_lag"] for r in turns), **counts}
        save(args.output / "jobs.json", jobs)
        save(args.output / "policy-events.json", history)
        save(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if model is not None:
            model.llm_engine.engine_core.shutdown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy", choices=["fcfs", "continuum"], required=True)
    p.add_argument("--implementation", choices=["upstream", "continuum"], default="continuum")
    p.add_argument("--allocator-gib", type=float, default=26., help="PyTorch allocator cap, not total GPU memory cap")
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--programs", type=int, help="默认使用数据文件中全部程序；显式设置才取前 N 条")
    p.add_argument("--max-turns", type=int, help="默认保留完整程序；显式设置才截取前 N 轮")
    p.add_argument("--jps", type=float, default=4.)
    p.add_argument("--kv-gib", type=float, default=2.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=600.)
    p.add_argument("--expected-uuid")
    args = p.parse_args()
    positive = [args.jps, args.kv_gib, args.timeout, args.max_context]
    positive.extend(x for x in (args.programs, args.max_turns) if x is not None)
    if min(positive) <= 0:
        p.error("workload, memory and timeout values must be positive")
    if args.implementation == "upstream" and args.policy != "fcfs":
        p.error("Upstream comparison uses original FCFS policy")
    run(args)


if __name__ == "__main__":
    main()
