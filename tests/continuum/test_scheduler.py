"""Real Scheduler + real KVCacheManager tests, no model weights/GPU kernels."""
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils import sha256
from vllm.v1.core.continuum_policy import ReconstructionCost, program_key
from vllm.v1.core.estimate_with_func import (ToolCallEstimator, ToolCallParser)
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher, init_none_hash)
from vllm.v1.core.sched.request_queue import (ContinuumRequestQueue,
                                              SchedulingPolicy)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.outputs import ModelRunnerOutput


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return "```bash\nls -l\n```" if 1 in tokens else "All done."


def estimator():
    return ToolCallEstimator(ReconstructionCost("prefill", 256, (0, 0, 100)),
                             tokenizer=Tokenizer())


def request(req_id, job="a", tokens=None, arrival=None):
    return Request(request_id=req_id, job_id=job,
                   prompt_token_ids=tokens if tokens is not None else [8] * 32,
                   sampling_params=SamplingParams(max_tokens=64),
                   arrival_time=arrival,
                   block_hasher=get_request_block_hasher(16, sha256))


def scheduler(blocks=32, budget=256, policy="continuum", chunked=True):
    init_none_hash(sha256)
    config = NS(
        scheduler_config=NS(policy=policy, max_num_seqs=16,
                            max_num_batched_tokens=budget, max_model_len=256,
                            long_prefill_token_threshold=0,
                            chunked_prefill_enabled=chunked),
        cache_config=NS(num_gpu_blocks=blocks, block_size=16,
                        enable_prefix_caching=True),
        parallel_config=NS(data_parallel_rank=0, decode_context_parallel_size=1,
                           pipeline_parallel_size=1),
        model_config=NS(is_encoder_decoder=False), lora_config=None,
        kv_events_config=None, kv_transfer_config=None, speculative_config=None)
    kv = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(16, 1, 1,
                                                     torch.float32, False))])
    with patch("vllm.v1.core.sched.scheduler.compute_encoder_budget",
               return_value=(0, 0)), patch.object(ToolCallEstimator, "from_config",
                                                 return_value=estimator()):
        structured = Mock()
        structured.should_advance.return_value = False
        return Scheduler(config, kv, structured)


def finish(s, req, tool=True, status=RequestStatus.FINISHED_STOPPED):
    req.append_output_token_ids(1 if tool else 2)
    s.finish_requests(req.request_id, status)


class HandlerTests(unittest.TestCase):
    def setUp(self):
        init_none_hash(sha256)

    def test_parser_supported_formats_and_terminal(self):
        parser = ToolCallParser()
        for text in ['{"name":"search","arguments":{}}',
                     '<tool_call>{"name":"search","arguments":{}}</tool_call>',
                     '{"tool_calls":[{"function":{"name":"search","arguments":"{}"}}]}']:
            self.assertEqual(parser.parse(text), "search")
        self.assertEqual(parser.parse("```bash\nls -l\n```"), "ls")
        self.assertIsNone(parser.parse("```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"))
        self.assertIsNone(parser.parse("```bash\nls\n```\n```bash\npwd\n```"))
        self.assertIsNone(parser.parse('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]'))

    def test_tool_duration_uses_server_arrival_not_handler_time(self):
        e = estimator()
        first = request("1", arrival=90.)
        e.request_arrives(first)
        first.append_output_token_ids(1)
        first.status = RequestStatus.FINISHED_STOPPED
        with patch("time.time", return_value=100.):
            e.request_finished(first)
        second = request("2", arrival=100.5)
        with patch("time.time", return_value=109.):
            e.request_arrives(second)
        self.assertEqual(e.policy.tool_durations["ls"], [.5])
        self.assertEqual(list(e.policy.completed_lengths), [])

    def test_final_and_abort_cleanup_without_future_steps(self):
        e = estimator()
        for status in [RequestStatus.FINISHED_ABORTED,
                       RequestStatus.FINISHED_LENGTH_CAPPED]:
            req = request(str(status))
            e.request_arrives(req)
            req.status = status
            e.request_finished(req)
            self.assertFalse(e.programs)
            self.assertEqual(list(e.policy.completed_lengths), [])
        req = request("final")
        e.request_arrives(req)
        req.status = RequestStatus.FINISHED_STOPPED
        e.request_finished(req)
        self.assertEqual(list(e.policy.completed_lengths), [1])

    def test_missing_ids_are_isolated_and_never_pinned(self):
        e = estimator()
        a, b = request("a", None), request("b", None)
        e.request_arrives(a)
        e.request_arrives(b)
        self.assertEqual(len(e.programs), 2)
        e.policy.observe_tool("ls", .1)
        a.status = RequestStatus.FINISHED_STOPPED
        a.append_output_token_ids(1)
        e.request_finished(a)
        self.assertEqual(e.set_up_pin(a), 0.)
        self.assertIn(program_key(b), e.programs)

    def test_overlap_rejected_explicitly(self):
        e = estimator()
        e.request_arrives(request("1"))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            e.request_arrives(request("2"))

    def test_evicted_wait_only(self):
        e = estimator()
        req = request("1", arrival=1.)
        e.request_arrives(req)
        with patch("time.time", return_value=3.):
            e.request_scheduled(req)
            e.mark_evicted(req, running=True)
        with patch("time.time", return_value=7.):
            e.request_scheduled(req)
        self.assertEqual(list(e.policy.queue_delays), [4.])


class QueueTests(unittest.TestCase):
    def setUp(self):
        init_none_hash(sha256)

    def test_pinned_then_program_fcfs(self):
        q = ContinuumRequestQueue()
        old = request("old", "old-job", arrival=1.)
        newer = request("newer", "new-job", arrival=2.)
        q.add_request(old)
        q.pop_request()
        old_return = request("return", "old-job", arrival=3.)
        q.add_request(newer)
        q.add_request(old_return)
        self.assertIs(q.peek_request(), old_return)
        self.assertIs(q.peek_request([(newer, 100.)]), newer)
        self.assertIs(q.pop_request(), old_return)

    def test_preempted_precedes_pinned_and_older_unpinned(self):
        # arXiv v6 §4.3 gives preempted status the highest priority.
        q = ContinuumRequestQueue()
        old = request("old", "old", arrival=1.)
        pinned = request("pinned", "pinned", arrival=2.)
        preempted = request("preempted", "preempted", arrival=3.)
        preempted.status = RequestStatus.PREEMPTED
        for req in (old, pinned, preempted):
            q.add_request(req)
        pins = [(pinned, 100.)]
        self.assertIs(q.peek_request(pins), preempted)
        self.assertIs(q.pop_request(pins), preempted)
        self.assertIs(q.pop_request(pins), pinned)
        self.assertIs(q.pop_request(pins), old)

    def test_preempted_category_orders_by_program_arrival(self):
        q = ContinuumRequestQueue()
        first = request("first", "a", arrival=1.)
        q.add_request(first)
        q.pop_request()
        returned = request("return", "a", arrival=10.)
        other = request("other", "b", arrival=2.)
        returned.status = other.status = RequestStatus.PREEMPTED
        q.add_request(other)
        q.add_request(returned)
        # TTL is only a priority dimension for the non-preempted category.
        self.assertIs(q.peek_request([(other, 100.)]), returned)


class SchedulerTests(unittest.TestCase):
    def test_real_scheduler_resumes_preempted_before_pinned_return(self):
        s = scheduler(budget=32)
        pin_owner = request("owner", "pinned", [5] * 32, arrival=1.)
        s.add_request(pin_owner)
        s.schedule()
        finish(s, pin_owner)
        follow = request("follow", "pinned", [5] * 32 + [7] * 32, arrival=2.)
        resumed = request("resumed", "newer", [9] * 32, arrival=3.)
        resumed.status = RequestStatus.PREEMPTED
        s.add_request(follow)
        s.add_request(resumed)
        result = s.schedule()
        self.assertEqual(set(result.num_scheduled_tokens), {"resumed"})
        self.assertEqual(result.scheduled_cached_reqs.req_ids, ["resumed"])
        self.assertEqual(result.scheduled_cached_reqs.resumed_from_preemption,
                         [True])
        self.assertEqual([r.request_id for r in s.waiting], ["follow"])

    def test_pin_expires_strictly_after_deadline(self):
        # Algorithm 1 explicitly uses current_time > P[id], not >=.
        s = scheduler()
        req = request("pin")
        s.add_request(req)
        s.schedule()
        finish(s, req)
        deadline = s.pinned_requests[0][1]
        with patch("time.time", return_value=deadline):
            s.unpin_requests_regular()
        self.assertEqual(len(s.pinned_requests), 1)
        with patch("time.time", return_value=deadline + .001):
            s.unpin_requests_regular()
        self.assertFalse(s.pinned_requests)

    def test_model_output_completion_enters_pin_lifecycle(self):
        s = scheduler()
        s.tool_call_estimator.policy.observe_tool("ls", 10.)
        req = request("inference")
        req.eos_token_id = 999
        s.add_request(req)
        for token in [1, 999]:
            scheduled = s.schedule()
            output = ModelRunnerOutput(
                req_ids=[req.request_id], req_id_to_index={req.request_id: 0},
                sampled_token_ids=[[token]], logprobs=None,
                prompt_logprobs_dict={}, pooler_output=[])
            s.update_from_output(scheduled, output)
        self.assertEqual(req.status, RequestStatus.FINISHED_STOPPED)
        self.assertEqual(len(s.pinned_requests), 1)
        self.assertEqual(s.get_num_unfinished_requests(), 0)
        self.assertNotIn(req.request_id, s.requests)

    def test_import_is_changed_source(self):
        import vllm.v1.core.sched.scheduler as source
        expected = Path(__file__).resolve().parents[2] / "vllm/v1/core/sched/scheduler.py"
        self.assertEqual(Path(source.__file__).resolve(), expected)
        if os.environ.get("CONTINUUM_SOURCE_ROOT"):
            self.assertEqual(expected.parents[4], Path(os.environ["CONTINUUM_SOURCE_ROOT"]))

    def test_fcfs_does_not_construct_tool_handler(self):
        s = scheduler(policy="fcfs")
        self.assertIsNone(s.tool_call_estimator)
        for name in ["1", "2"]:
            s.add_request(request(name, None))
        self.assertEqual(len(s.schedule().scheduled_new_reqs), 2)

    def test_return_transfers_pin_references_after_allocation(self):
        s = scheduler()
        s.tool_call_estimator.policy.observe_tool("ls", 10.)
        first = request("first")
        s.add_request(first)
        s.schedule()
        first_blocks = s.kv_cache_manager.get_blocks(first.request_id).blocks[0][:]
        finish(s, first)
        self.assertEqual(len(s.pinned_requests), 1)
        follow = request("follow", tokens=[8] * 32 + [1, 9, 9])
        s.add_request(follow)
        output = s.schedule()
        self.assertEqual(len(s.pinned_requests), 0)
        self.assertEqual(follow.num_cached_tokens, 32)
        for block in first_blocks:
            self.assertEqual(block.ref_cnt, 1)
        self.assertEqual(output.num_common_prefix_blocks, [0])
        finish(s, follow, tool=False)
        self.assertFalse(s.tool_call_estimator.programs)
        self.assertTrue(all(block.ref_cnt == 0 for block in first_blocks))

    def test_consecutive_expired_pins_and_waiting_protection(self):
        s = scheduler()
        s.tool_call_estimator.policy.observe_tool("ls", 10.)
        previous = []
        for i in range(3):
            req = request(f"old-{i}", str(i), [8 + i] * 32)
            s.add_request(req)
            previous.append(req)
        s.schedule()
        for req in previous:
            finish(s, req)
        s.pinned_requests[:] = [(req, 0.) for req, _ in s.pinned_requests]
        s.add_request(request("return", "1", [9] * 32 + [1, 2]))
        s.unpin_requests_regular()
        self.assertEqual([r.job_id for r, _ in s.pinned_requests], ["1"])

    def test_deadlock_retries_multiple_latest_programs_same_step(self):
        # 1 null block + 3 programs * 2 blocks fills the pool.
        s = scheduler(blocks=7)
        s.tool_call_estimator.policy.observe_tool("ls", 10.)
        old = []
        for i in range(3):
            req = request(f"old-{i}", str(i), [20 + i] * 32,
                          time.time() - 3 + i)
            old.append(req)
            s.add_request(req)
        s.schedule()
        for req in reversed(old):
            finish(s, req)
        order = []
        original = s.unpin_request
        def record(req, deadline, evicted=True):
            if evicted:
                order.append(req.job_id)
            return original(req, deadline, evicted)
        s.unpin_request = record
        incoming = request("large", "new", [77] * 64)
        s.add_request(incoming)
        output = s.schedule()
        self.assertIn("large", output.num_scheduled_tokens)
        self.assertEqual(order, ["2", "1"])

    def test_final_abort_frees_queued_return_and_previous_pin(self):
        s = scheduler()
        s.tool_call_estimator.policy.observe_tool("ls", 10.)
        old = request("old")
        s.add_request(old)
        s.schedule()
        finish(s, old)
        follow = request("follow")
        s.add_request(follow)
        s.finish_requests("follow", RequestStatus.FINISHED_ABORTED)
        self.assertFalse(s.pinned_requests)
        self.assertFalse(s.tool_call_estimator.programs)

    def test_skip_removes_selected_request_not_other_priority_head(self):
        s = scheduler(budget=16, chunked=False)
        s.tool_call_estimator.policy.observe_tool("ls", 10.)
        # Remember older program first, but leave it out of waiting for now.
        old = request("old", "old", [4] * 8, time.time() - 10)
        s.add_request(old)
        s.schedule()
        finish(s, old, tool=False)
        # A later pinned program's return is too large without chunking.
        pin = request("pin", "pinned", [5] * 8)
        s.add_request(pin)
        s.schedule()
        finish(s, pin)
        a = request("a", "older-unpinned", [7] * 8, time.time() - 5)
        b = request("b", "pinned", [5] * 32)
        s.add_request(a)
        s.add_request(b)
        output = s.schedule()
        self.assertEqual(set(output.num_scheduled_tokens), {"a"})
        self.assertEqual([r.request_id for r in s.waiting], ["b"])

    def test_running_single_request_preemption_does_not_crash(self):
        s = scheduler(blocks=3)
        req = request("single", tokens=[7] * 32)
        s.add_request(req)
        s.schedule()
        req.append_output_token_ids(9)
        output = s.schedule()
        self.assertEqual(output.total_num_scheduled_tokens, 0)
        self.assertEqual(req.status, RequestStatus.PREEMPTED)

    def test_delayed_free_cannot_repin_old_generation(self):
        s = scheduler()
        s.tool_call_estimator.policy.observe_tool("ls", 10.)
        first = request("first")
        s.add_request(first)
        s.schedule()
        with patch.object(s, "_connector_finished", return_value=(True, None)):
            finish(s, first)
        follow = request("follow", tokens=[8] * 32 + [1, 9])
        s.add_request(follow)
        s.schedule()
        finish(s, follow)
        self.assertEqual([r.request_id for r, _ in s.pinned_requests], ["follow"])
        s._free_blocks(first)
        self.assertEqual([r.request_id for r, _ in s.pinned_requests], ["follow"])

    def test_preempt_already_scheduled_request_restores_bookkeeping(self):
        s = scheduler(blocks=8)
        reqs = []
        for i, arrival in enumerate([3., 1., 2.]):
            req = request(str(i), str(i), [8 + i] * 32, arrival)
            reqs.append(req)
            s.add_request(req)
        s.schedule()
        # Program FCFS initially sorts them. Exercise a changed running order,
        # as can happen after previous chunked-prefill/preemption cycles.
        s.running[:] = reqs
        for req in reqs:
            req.append_output_token_ids(9)
        output = s.schedule()
        self.assertEqual(set(output.num_scheduled_tokens), {"1", "2"})
        self.assertEqual(output.total_num_scheduled_tokens, 2)
        self.assertEqual(reqs[0].status, RequestStatus.PREEMPTED)
        self.assertEqual(set(output.scheduled_cached_reqs.req_ids), {"1", "2"})

    def test_duplicate_abort_during_delayed_free_does_not_finish_twice(self):
        s = scheduler()
        req = request("first")
        s.add_request(req)
        s.schedule()
        with patch.object(s, "_connector_finished", return_value=(True, None)):
            finish(s, req, tool=False)
        s.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
        self.assertEqual(list(s.tool_call_estimator.policy.completed_lengths), [1])
        s._free_blocks(req)
        self.assertNotIn(req.request_id, s.requests)

    def test_unpin_own_partial_prefix_then_retry(self):
        s = scheduler(blocks=4)
        s.tool_call_estimator.policy.observe_tool("ls", 10.)
        old = request("old", tokens=[8] * 48)
        s.add_request(old)
        s.schedule()
        finish(s, old)
        follow = request("follow", tokens=[8] * 32 + [9] * 16)
        s.add_request(follow)
        output = s.schedule()
        self.assertIn("follow", output.num_scheduled_tokens)
        self.assertEqual(follow.num_cached_tokens, 32)
        self.assertFalse(s.pinned_requests)
        self.assertTrue(all(b.ref_cnt == 1 for b in
                            s.kv_cache_manager.get_blocks("follow").blocks[0]))


if __name__ == "__main__":
    unittest.main()
