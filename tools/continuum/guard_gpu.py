"""Sample GPU usage; abort ONLY the launched process group on contention.

Not a hardware memory/compute partition. Limits are sampled, not instantaneous.
Does not stop, reprioritize or modify any existing process or device setting.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def smi(arguments):
    return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=8)


def snapshot(uuid):
    device = smi([f"--id={uuid}", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
                  "--format=csv,noheader,nounits"]).strip().split(",")
    index, total, used, free, utilization = map(lambda x: int(x.strip()), device)
    processes = {}
    for line in smi([f"--id={uuid}", "--query-compute-apps=pid,used_gpu_memory",
                     "--format=csv,noheader,nounits"]).splitlines():
        pid, memory = line.split(",")
        # N/A means monitoring cannot enforce the requested memory guard.
        processes[int(pid)] = int(memory.strip())
    activity = {}
    for line in smi(["pmon", "-i", str(index), "-c", "1", "-s", "u"]).splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].isdigit() and fields[1].isdigit():
            activity[int(fields[1])] = {
                "sm": None if fields[3] == "-" else int(fields[3]),
                "memory": None if fields[4] == "-" else int(fields[4])}
    return {"time": time.time(), "index": index, "total_mib": total,
            "used_mib": used, "free_mib": free, "gpu_utilization": utilization,
            "processes": processes, "activity": activity}


def violations(sample, baseline, own_pids, max_own_mib, min_free_mib, growth_mib):
    foreign = {pid: mib for pid, mib in sample["processes"].items() if pid not in own_pids}
    own = sum(mib for pid, mib in sample["processes"].items() if pid in own_pids)
    reasons = []
    if own > max_own_mib:
        reasons.append("own_memory_cap_exceeded")
    if sample["free_mib"] < min_free_mib:
        reasons.append("free_memory_below_floor")
    if set(foreign) - set(baseline["processes"]):
        reasons.append("new_foreign_gpu_process")
    if any(mib - baseline["processes"].get(pid, 0) > growth_mib for pid, mib in foreign.items()):
        reasons.append("foreign_memory_growth")
    for pid in foreign:
        utilization = sample["activity"].get(pid, {})
        if any((utilization.get(key) or 0) > 0 for key in ("sm", "memory")):
            reasons.append("foreign_compute_activity")
    return reasons, own


def stop_owned_group(child):
    if child.poll() is not None:
        return
    if os.getpgid(child.pid) != child.pid:
        raise RuntimeError("Unexpected process group; refusing to signal")
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # The unreaped group leader remains our child, not an arbitrary GPU PID.
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uuid", required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--monitor", type=Path, required=True)
    p.add_argument("--max-own-gib", type=float, default=28)
    p.add_argument("--min-free-gib", type=float, default=100)
    p.add_argument("--foreign-growth-mib", type=int, default=512)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or min(args.max_own_gib, args.min_free_gib, args.timeout) <= 0:
        p.error("positive budgets and a command are required")
    baseline = snapshot(args.uuid)
    reasons, _ = violations(baseline, baseline, set(), args.max_own_gib * 1024,
                            args.min_free_gib * 1024, args.foreign_growth_mib)
    if reasons:
        raise SystemExit(f"GPU not suitable before launch: {reasons}")
    child = None
    exit_reason, max_own = None, 0
    with args.log.open("x") as output, args.monitor.open("x") as monitor:
        def record(row):
            monitor.write(json.dumps(row) + "\n")
            monitor.flush()
        record({"kind": "baseline", "uuid": args.uuid, "command": command,
                "budgets": vars(args) | {"log": str(args.log), "monitor": str(args.monitor)}, **baseline})
        try:
            # vLLM 0.10.2 内部 NVML 映射只接受数字序号，不接受 UUID。
            # 先用 UUID 查序号，再固定 PCI 顺序；模型加载前还会核对 torch UUID。
            child_env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                             CUDA_VISIBLE_DEVICES=str(baseline["index"]))
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=child_env)
            launched = time.monotonic()
            while child.poll() is None:
                sample = snapshot(args.uuid)
                own_pids = set()
                for pid in sample["processes"]:
                    try:
                        if os.getpgid(pid) == child.pid:
                            own_pids.add(pid)
                    except ProcessLookupError:
                        pass
                reasons, own = violations(sample, baseline, own_pids, args.max_own_gib * 1024,
                                           args.min_free_gib * 1024, args.foreign_growth_mib)
                max_own = max(max_own, own)
                record({"kind": "sample", "own_pids": sorted(own_pids), "own_mib": own,
                        "violations": reasons, **sample})
                if time.monotonic() - launched > args.timeout:
                    reasons.append("wall_time_budget")
                if reasons:
                    exit_reason = reasons
                    stop_owned_group(child)
                    break
                time.sleep(1)
        except BaseException as error:
            exit_reason = [type(error).__name__, str(error)]
            if child is not None:
                stop_owned_group(child)
            raise
        finally:
            record({"kind": "end", "returncode": child.poll() if child else None,
                    "exit_reason": exit_reason, "max_own_mib": max_own, "time": time.time()})
        raise SystemExit(125 if exit_reason else child.returncode)


if __name__ == "__main__":
    main()
