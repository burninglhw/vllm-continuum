"""小验证集工具的 CPU 测试；不加载 GPU、不访问外部 API。"""
import gzip
import io
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.continuum.trace_dataset import (DataOnlyUnpickler, digest, load_replay,
                                          program_stats, protocol_issues,
                                          select_programs, summarize)
from tools.continuum.verify_gpu import load_workload, scheduled_prefill_tokens


def program(name, turns=3, gap=1., accepted=True):
    return {"instance_id": name, "steps": [
        {"prompt": [1] * (16 + i), "output": [41 if i + 1 < turns else 42, 99],
         "tool": "ls" if i + 1 < turns else None, "gap": gap,
         "accepted": accepted, "turn": i + 1, "attempt": i + 1}
        for i in range(turns)]}


class TraceDatasetTests(unittest.TestCase):
    def test_selection_preserves_whole_programs_and_source_order(self):
        original = [program("slow", gap=9), program("medium", gap=2), program("fast", gap=1)]
        selected, info = select_programs(original, count=2, max_context=64,
                                          min_turns=2, max_turns=4)
        self.assertEqual([p["instance_id"] for p in selected], ["medium", "fast"])
        self.assertIs(selected[0], original[1])
        self.assertEqual(len(selected[0]["steps"]), 3)
        self.assertEqual(info["eligible_count"], 3)

    def test_filters_are_explicit(self):
        long = program("long")
        long["steps"][-1]["prompt"] = [1] * 100
        selected, info = select_programs([long, program("recovery", accepted=False), program("ok")],
            count=1, max_context=64, min_turns=2, max_turns=4)
        self.assertEqual(selected[0]["instance_id"], "ok")
        self.assertEqual(len(info["rejected"]), 2)

    def test_not_enough_traces_fails_without_truncating(self):
        with self.assertRaisesRegex(ValueError, "Only 1"):
            select_programs([program("a")], count=2, max_context=64, min_turns=2, max_turns=4)

    def test_pickle_globals_are_forbidden(self):
        with self.assertRaises(pickle.UnpicklingError):
            DataOnlyUnpickler(io.BytesIO(pickle.dumps(eval))).load()

    def test_protocol_and_eos_checks(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: "tool" if ids[0] == 41 else "final")
        parser = SimpleNamespace(parse=lambda text: "ls" if text == "tool" else None)
        data = program("ok")
        self.assertEqual(protocol_issues(data, tokenizer, parser, {99}), [])
        data["steps"][0]["output"] = [42, 99]
        self.assertIn("nonfinal_output_looks_terminal_to_current_parser",
                      protocol_issues(data, tokenizer, parser, {99}))

    def test_stats_and_final_tool_gap_preserved(self):
        data = program("a", turns=3, gap=2)
        self.assertEqual(program_stats(data)["tool_seconds"], 6)
        self.assertEqual(summarize([data])["responses"], 3)

    def test_prefill_counter_uses_pre_schedule_position(self):
        self.assertEqual(scheduled_prefill_tokens(100, 40, 40), 40)
        self.assertEqual(scheduled_prefill_tokens(100, 105, 10), 5)
        self.assertEqual(scheduled_prefill_tokens(100, 106, 1), 0)

    def test_manifest_and_no_implicit_prefix_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            hashes = {}
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (model / name).write_text("{}")
                hashes[name] = digest(model / name)
            path = root / "replay.pkl.gz"
            with gzip.open(path, "wb") as stream:
                pickle.dump([program("a", turns=12)], stream)
            manifest = {"replay_sha256": digest(path), "tokenizer_file_hashes": hashes}
            (root / "manifest.json").write_text(json.dumps(manifest))
            args = SimpleNamespace(data=path, model=str(model), programs=None,
                                   max_turns=None, max_context=64)
            data, _ = load_workload(args)
            self.assertEqual(len(data[0]["steps"]), 12)
            args.max_context = 20
            with self.assertRaisesRegex(ValueError, "Context overflow"):
                load_workload(args)
            manifest["replay_sha256"] = "not-the-real-hash"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_replay(path)


if __name__ == "__main__":
    unittest.main()
