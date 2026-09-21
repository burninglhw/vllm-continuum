"""查看真实 SWE trace，并选取完整的小规模 GPU 回放集；不运行模型或调用 API。

raw traces.jsonl：每行一个完整 agent 程序，包含多轮 messages/response/tool。
replay.pkl.gz：同一程序已被 Llama tokenizer 编码，供真实 GPU 前向回放使用。
筛选只按轮次、长度、协议兼容性和工具等待成本，不读取任何性能对比结果。
"""
import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import statistics


class DataOnlyUnpickler(pickle.Unpickler):
    # pickle 通常能执行任意 Python；这里只接受普通容器/数值，不允许加载全局类。
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Replay data must not contain globals")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    # 不覆盖历史结果；想重新生成时应换一个输出目录。
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def load_replay(path):
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if digest(path) != manifest["replay_sha256"]:
        raise ValueError("Replay SHA-256 differs from its manifest")
    with gzip.open(path, "rb") as stream:
        programs = DataOnlyUnpickler(stream).load()
    if not isinstance(programs, list) or not programs:
        raise ValueError("Expected a non-empty list of programs")
    ids = [p["instance_id"] for p in programs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate instance_id")
    for program in programs:
        if not program["steps"]:
            raise ValueError("Empty program")
    return programs, manifest


def program_stats(program):
    steps = program["steps"]
    return {
        "instance_id": program["instance_id"], "responses": len(steps),
        "max_total_tokens": max(len(s["prompt"]) + len(s["output"]) for s in steps),
        # EOS 不能恰好落在长度封顶边界；为正常 STOPPED 留一 token 余量。
        "required_context": max(len(s["prompt"]) + len(s["output"]) + 1 for s in steps),
        "prompt_tokens": sum(len(s["prompt"]) for s in steps),
        "output_tokens": sum(len(s["output"]) for s in steps),
        "tool_seconds": sum(s["gap"] for s in steps),
        "tool_calls": sum(s["tool"] is not None for s in steps),
        "rejected_or_recovery_responses": sum(not s["accepted"] for s in steps),
    }


def select_programs(programs, *, count, max_context, min_turns, max_turns,
                    compatibility=lambda p: []):
    eligible, rejected = [], []
    for program in programs:
        stats = program_stats(program)
        reasons = []
        if not min_turns <= stats["responses"] <= max_turns:
            reasons.append("outside_turn_range")
        if stats["required_context"] > max_context:
            reasons.append("outside_context")
        if stats["rejected_or_recovery_responses"]:
            reasons.append("contains_protocol_recovery")
        if not reasons:
            reasons.extend(compatibility(program))
        if reasons:
            rejected.append({"instance_id": program["instance_id"], "reasons": reasons})
        else:
            eligible.append((program, stats))
    # 快速调试集偏向工具等待较短；它不是代表完整长尾分布的随机样本。
    eligible.sort(key=lambda pair: (pair[1]["tool_seconds"],
                                   pair[1]["prompt_tokens"], pair[0]["instance_id"]))
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible complete traces for count={count}; "
                         "choose a smaller count or explicit wider limits")
    chosen = {p["instance_id"] for p, _ in eligible[:count]}
    # 选中后恢复原数据集顺序。两种调度器用同一列表，seed 仅改变初始到达间隔。
    selected = [p for p in programs if p["instance_id"] in chosen]
    return selected, {"eligible_count": len(eligible), "rejected": rejected,
                      "eligible_ranked": [s for _, s in eligible]}


def tokenizer_for(model, manifest):
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if digest(Path(model) / name) != manifest["tokenizer_file_hashes"][name]:
            raise ValueError(f"Tokenizer mismatch: {name}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, local_files_only=True,
                                        trust_remote_code=False)


def protocol_issues(program, tokenizer, parser, eos_ids):
    issues = set()
    for index, step in enumerate(program["steps"]):
        output = step["output"]
        if not output or output[-1] not in eos_ids or set(output[:-1]) & eos_ids:
            issues.add("not_exactly_one_terminal_eos")
        tool = parser.parse(tokenizer.decode(output, skip_special_tokens=True))
        if index + 1 < len(program["steps"]) and not tool:
            issues.add("nonfinal_output_looks_terminal_to_current_parser")
        if index + 1 < len(program["steps"]) and tool != step["tool"]:
            issues.add("parsed_tool_differs_from_trace_label")
        if index + 1 == len(program["steps"]) and tool is not None:
            issues.add("final_output_not_recognized_as_terminal")
    return sorted(issues)


def load_raw_selected(path, ids, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError("Raw trace SHA-256 differs from source replay manifest")
    selected = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            if trace["instance_id"] in ids:
                selected[trace["instance_id"]] = trace
    if set(selected) != set(ids):
        raise ValueError("Selected IDs are missing from raw traces")
    return selected


def summarize(programs):
    rows = [program_stats(p) for p in programs]
    return {"programs": len(rows), "responses": sum(r["responses"] for r in rows),
            "mean_responses_per_program": statistics.mean(r["responses"] for r in rows),
            "std_responses_per_program": statistics.pstdev(r["responses"] for r in rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "max_context_required": max(r["required_context"] for r in rows),
            "tool_seconds": sum(r["tool_seconds"] for r in rows), "records": rows}


def build(args):
    programs, source_manifest = load_replay(args.data)
    tokenizer = tokenizer_for(args.model, source_manifest)
    # 用真正的新解析器检查兼容性；这只加载 tokenizer/源码，不加载模型权重。
    from vllm.v1.core.estimate_with_func import ToolCallParser
    eos_ids = json.loads((Path(args.model) / "config.json").read_text())["eos_token_id"]
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    selected, diagnostics = select_programs(
        programs, count=args.count, max_context=args.max_context,
        min_turns=args.min_turns, max_turns=args.max_turns,
        compatibility=lambda p: protocol_issues(p, tokenizer, ToolCallParser(), eos_ids))
    raw = load_raw_selected(args.raw_traces, {p["instance_id"] for p in selected},
                            source_manifest["trace_sha256"])
    for program in selected:
        events = [e for e in raw[program["instance_id"]]["events"] if e["response"] is not None]
        if len(events) != len(program["steps"]):
            raise ValueError("Raw/tokenized response count mismatch")
        for event, step in zip(events, program["steps"]):
            # decode 可能规范化空格，不能用反解文本相等代替 token 来源校验。
            # 重走原 prepare_replay 的聊天模板，逐个核对全部输入/输出 token。
            messages = event["request"]["messages"]
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            completed = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": event["response"]["content"]}],
                tokenize=True, add_generation_prompt=False)
            if (prompt != step["prompt"] or completed[:len(prompt)] != prompt
                    or completed[len(prompt):] != step["output"]):
                raise ValueError(f"Raw/template token mismatch: {program['instance_id']} turn={step['turn']}")
    args.output.mkdir(parents=True, exist_ok=False)
    # 原始轨迹/所有轮次完整保留，不删 reasoning、不改命令、不压缩工具时长。
    # GPU 回放本来只使用可见响应；raw 文件保留源记录，供人阅读和溯源。
    with (args.output / "traces.jsonl").open("x", encoding="utf-8") as stream:
        for program in selected:
            stream.write(json.dumps(raw[program["instance_id"]], ensure_ascii=False) + "\n")
    with (args.output / "replay.pkl.gz").open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            pickle.dump(selected, compressed, protocol=5)
    summary = summarize(selected)
    selection = {
        "purpose": "fast functional/debug replay; NOT representative paper performance",
        "count": args.count, "min_turns": args.min_turns, "max_turns": args.max_turns,
        "max_context": args.max_context,
        "rule": "complete traces; no protocol recovery; current parser/EOS compatible; "
                "rank by tool seconds, prompt tokens, ID; retain original source order",
        "selection_seed": None, "arrival_seed_is_a_separate_run_parameter": True,
        "source_replay": str(args.data.resolve()), "source_traces": str(args.raw_traces.resolve()),
        "source_replay_sha256": source_manifest["replay_sha256"],
        "source_trace_sha256": source_manifest["trace_sha256"],
        "source_programs": len(programs), "source_responses": sum(len(p["steps"]) for p in programs),
        "source_not_modified": True, "whole_programs_preserved": True,
        "all_selected_prompt_and_output_tokens_reencoded_from_raw": True,
        "paper_reference": "Continuum Table 1 / section 2.2: SWE turns mean=10.9, std=2.1; "
                           "our response-attempt count is an approximation, not identical semantics",
        **summary, **diagnostics,
    }
    write_json(args.output / "selection.json", selection)
    write_json(args.output / "manifest.json", {
        "schema_version": 1, **summary,
        "replay_sha256": digest(args.output / "replay.pkl.gz"),
        "trace_sha256": digest(args.output / "traces.jsonl"),
        "source_replay_sha256": source_manifest["replay_sha256"],
        "tokenizer_file_hashes": source_manifest["tokenizer_file_hashes"],
        "whole_programs": True, "reasoning_included": False,
        "not_a_paper_performance_reproduction": True,
    })
    write_json(args.output / "lengths.json", [{"instance_id": p["instance_id"], "steps": [
        {**{k: v for k, v in s.items() if k not in ("prompt", "output")},
         "input_tokens": len(s["prompt"]), "output_tokens": len(s["output"])}
        for s in p["steps"]]} for p in selected])
    write_json(args.output / "example-response.json", decoded_step(selected[0], 1, tokenizer, 0))
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


def decoded_step(program, number, tokenizer, limit):
    if not 1 <= number <= len(program["steps"]):
        raise ValueError(f"step must be between 1 and {len(program['steps'])}")
    step = program["steps"][number - 1]
    prompt = tokenizer.decode(step["prompt"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    response = tokenizer.decode(step["output"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"instance_id": program["instance_id"], "response_index_1based": number,
            "response_count": len(program["steps"]), "trace_turn": step["turn"],
            "trace_attempt": step["attempt"], "prompt_tokens": len(step["prompt"]),
            "output_tokens": len(step["output"]), "tool_label": step["tool"],
            "tool_gap_seconds": step["gap"], "accepted_into_context": step["accepted"],
            "prompt_text": prompt[:limit] if limit else prompt,
            "response_text": response[:limit] if limit else response,
            "text_truncated": bool(limit and (len(prompt) > limit or len(response) > limit)),
            "prompt_token_ids_head": step["prompt"][:16],
            "output_token_ids_head": step["output"][:16]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("summary", "show", "build"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", type=Path, required=True)
        if name in ("show", "build"):
            sub.add_argument("--model", required=True, help="本地 tokenizer 所在模型目录；不加载权重")
        if name == "show":
            sub.add_argument("--instance")
            sub.add_argument("--step", type=int, default=1, help="第几次响应，从 1 开始，不是 trace turn")
            sub.add_argument("--limit", type=int, default=1200, help="每段文本最多字符；0 显示全文")
            sub.add_argument("--output", type=Path)
        if name == "build":
            sub.add_argument("--raw-traces", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--count", type=int, default=5)
            sub.add_argument("--min-turns", type=int, default=8)
            sub.add_argument("--max-turns", type=int, default=14)
            sub.add_argument("--max-context", type=int, default=8192)
    args = parser.parse_args()
    if args.command == "build":
        if min(args.count, args.min_turns, args.max_context) < 1 or args.max_turns < args.min_turns:
            parser.error("invalid count/turn/context bounds")
        build(args)
        return
    programs, manifest = load_replay(args.data)
    if args.command == "summary":
        result = summarize(programs)
    else:
        if args.limit < 0:
            parser.error("limit must be non-negative")
        selected = [p for p in programs if p["instance_id"] == args.instance] if args.instance else programs[:1]
        if not selected:
            parser.error("instance not found")
        result = decoded_step(selected[0], args.step, tokenizer_for(args.model, manifest), args.limit)
        if args.output:
            write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
