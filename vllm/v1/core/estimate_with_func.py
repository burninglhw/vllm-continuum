# SPDX-License-Identifier: Apache-2.0
# 中文导读：连接“请求生命周期”和“纯策略模型”的服务端适配层。
# 到达时学上一轮工具耗时；完成时解析这轮输出并求 TTL；不接收未来工具时长。
"""Server-side tool-call handler for paper Continuum, not an Elastic policy."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.continuum_policy import (ContinuumPolicy, ReconstructionCost,
                                           program_key)
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

class Continuum_Recorder:
    # 日志按程序分组，记录到达、调度、完成、pin/unpin 和 TTL 决策输入。
    # shutdown 时写出 scheduler_timestamps；这不是 DeepSeek 原始数据集。
    """Public repository timeline format, plus auditable TTL decision inputs."""

    def __init__(self):
        self.job_id_to_history = {}
        self.scheduling_times = []

    def record(self, request, **event):
        key = (str(request.job_id) if request.job_id is not None
               else f"request:{request.request_id}")
        self.job_id_to_history.setdefault(key, []).append(event)

    def print_history(self):
        output_dir = os.environ.get("RUN_OUTPUT_DIR", "./continuum_exp")
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, "scheduler_timestamps")
        with open(final_path + ".tmp", "w") as f:
            json.dump(self.job_id_to_history, f, indent=2)
        os.replace(final_path + ".tmp", final_path)

    def request_arrives(self, request):
        self.record(request, Request_arrival_time=request.arrival_time)

    def request_finished(self, request):
        self.record(request, Request_departure_time=time.time())

    def request_evicted_from_running_queue(self, request):
        self.record(request, Request_evicted_from_running_queue_time=time.time())

    def request_pinned(self, request):
        self.record(request, pinned_time=time.time(),
                    deadline=request.continuum_pin_deadline)

    def request_unpinned(self, request):
        self.record(request, unpinned_time=time.time())

    def request_waiting_to_running(self, request, prompt_length, hit_length=0):
        self.record(request, waiting_to_running=time.time(),
                    prompt_length=prompt_length, hit_length=hit_length)

    def request_evicted_to_running(self, request, prompt_length, hit_length):
        self.record(request, evicted_to_running=time.time(),
                    prompt_length=prompt_length, hit_length=hit_length)


class ToolCallParser:
    # 仅识别约定的单个 bash/JSON 调用。返回的是工具名（如 ls、python），
    # 不执行命令；真正工具执行由客户端/agent 完成。无法识别会返回 None。
    """Appendix D bash parser plus a single JSON function invocation.

    Not a universal model-specific OpenAI tool parser. Parallel calls and
    arbitrary Python/function-string syntaxes are deliberately unsupported.
    """

    def parse(self, text: str) -> Optional[str]:
        actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)
        if actions:
            if len(actions) != 1:
                return None
            action = actions[0].strip()
            # mini-SWE-agent terminates on these shell output sentinels.
            if re.match(r"^echo\s+['\"]?(?:COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|"
                        r"MINI_SWE_AGENT_FINAL_OUTPUT)(?:['\"]?)(?:\s|$)", action):
                return None
            words = action.split()
            return words[0] if words else None
        payload = text.strip()
        match = re.fullmatch(r"(?:```json\s*\n|<tool_call>\s*)(.*?)"
                             r"(?:\n```|\s*</tool_call>)", payload, re.DOTALL)
        if match:
            payload = match.group(1)
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            return None
        if isinstance(obj, dict) and "tool_calls" in obj:
            obj = obj["tool_calls"]
        if isinstance(obj, list):
            if len(obj) != 1:
                return None
            obj = obj[0]
        if isinstance(obj, dict) and "function" in obj:
            obj = obj["function"]
        if isinstance(obj, dict) and "arguments" in obj:
            name = obj.get("name")
            if isinstance(name, str) and name:
                return name
        return None


@dataclass
class ProgramState:
    # 一个程序可以依次经历多轮 Request，但这里只允许一轮处于 active。
    # pending 保存“哪轮完成、调用了哪个工具、何时完成”，等待下一轮回来。
    first_arrival: float
    completed_steps: int = 0
    active_request: Optional[str] = None
    active_arrival: float = 0.
    pending: Optional[tuple[str, str, float]] = None
    evicted: bool = False


class ToolCallEstimator:
    def __init__(self, cost: ReconstructionCost, tokenizer=None,
                 model_name: Optional[str] = None, tokenizer_mode="auto",
                 trust_remote_code=False, tokenizer_revision=None,
                 parser=None, queue_window=256, program_window=256,
                 recorder=None):
        self.policy = ContinuumPolicy(cost, queue_window, program_window)
        self.programs = {}
        self.waiting_since = {}
        self.recorder = recorder
        if tokenizer is None and model_name is not None:
            from vllm.transformers_utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(
                tokenizer_name=model_name, tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code, revision=tokenizer_revision)
        if tokenizer is None:
            raise ValueError("Continuum needs a tokenizer to parse tool output")
        self.tokenizer = tokenizer
        self.parser = parser or ToolCallParser()

    @classmethod
    def from_config(cls, config, recorder=None):
        # 从 --additional-config 的 continuum 字段读取实测 profile 与窗口。
        # APC/同步调度是当前实现前提；配置不符就报错，不静默退回旧两秒策略。
        options = config.additional_config.get("continuum", {})
        allowed = {"profile_path", "queue_window", "program_window"}
        if set(options) - allowed:
            raise ValueError(f"Unknown Continuum options: {set(options) - allowed}")
        if "profile_path" not in options:
            raise ValueError("Paper Continuum requires additional_config.continuum."
                             "profile_path; see docs/continuum-paper.md")
        if not config.cache_config.enable_prefix_caching:
            raise ValueError("Continuum requires --enable-prefix-caching")
        if config.scheduler_config.async_scheduling:
            raise ValueError("Paper Continuum currently requires sync scheduling")
        model = config.model_config
        cost = ReconstructionCost.from_file(
            options["profile_path"], model.model,
            config.scheduler_config.max_model_len,
            signature={"dtype": str(model.dtype),
                       "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
                       "enforce_eager": model.enforce_eager})
        if (cost.mode == "reload") != (config.kv_transfer_config is not None):
            raise ValueError("reload profile requires KV offload connector; "
                             "prefill profile requires no connector")
        return cls(cost=cost, model_name=model.tokenizer,
                   tokenizer_mode=model.tokenizer_mode,
                   trust_remote_code=model.trust_remote_code,
                   tokenizer_revision=model.tokenizer_revision,
                   queue_window=options.get("queue_window", 256),
                   program_window=options.get("program_window", 256),
                   recorder=recorder)

    def request_arrives(self, request: Request):
        # 同 job_id 的下一轮到达，才知道上一轮实际等了多久。
        # 用请求的到达时间而非本函数运行时间，避免把入口处理延迟当工具耗时。
        key = program_key(request)
        state = self.programs.setdefault(key, ProgramState(request.arrival_time))
        if state.active_request is not None:
            raise ValueError("Continuum requires sequential requests and unique "
                             f"program IDs; overlapping program {key}")
        if state.pending is not None:
            _, tool, finished = state.pending
            self.policy.observe_tool(tool, request.arrival_time - finished)
            request.last_func_call = tool
            state.pending = None
        state.active_request = request.request_id
        state.active_arrival = request.arrival_time
        if state.evicted:
            self.waiting_since[request.request_id] = request.arrival_time

    def mark_evicted(self, request: Request, running=False):
        # 此处 evicted 表示“保护撤销/运行被抢占”，不保证 KV 已被物理覆盖。
        # 未覆盖的块仍可能 APC 命中；这与旧回放器按 LCP 缺失判断的口径不同。
        state = self.programs.get(program_key(request))
        if state is None:
            return
        state.evicted = True
        if state.active_request is not None:
            self.waiting_since.setdefault(
                state.active_request, time.time() if running else state.active_arrival)

    def request_scheduled(self, request: Request):
        # 只有之前标记为被回收的等待片段才训练队列代价；新请求普通排队不计。
        start = self.waiting_since.pop(request.request_id, None)
        if start is not None:
            self.policy.observe_evicted_wait(time.time() - start)
        self.programs[program_key(request)].evicted = False

    def request_finished(self, request: Request):
        # 一轮请求结束 != 整个程序结束。正常输出工具调用后，程序进入 pending。
        # 取消/长度封顶不继续 pin；解析不到工具暂按程序结束处理，这是当前边界：
        # 格式错误后继续重试的 agent 可能被分成多个统计片段。
        key = program_key(request)
        state = self.programs[key]
        finished = time.time()
        self.waiting_since.pop(request.request_id, None)
        state.active_request = None
        request.continuum_pin_deadline = 0.
        normal = request.status == RequestStatus.FINISHED_STOPPED
        tool = None
        if normal:
            try:
                text = self.tokenizer.decode(request.output_token_ids,
                                             skip_special_tokens=True)
                tool = self.parser.parse(text)
            except Exception:
                # Parsing must never strand GPU references or poison eta.
                logger.exception("Continuum could not parse request %s",
                                 request.request_id)
                normal = False
        # Terminal metadata is consumed ONLY after completion, never priority.
        final = (not normal or not tool or request.is_last_step is True
                 or request.job_id is None)
        # 最终标记只在这里消费，不用于队列优先级或抢占候选排序。
        request.this_func_call = None if final else tool
        request.continuum_program_finished = final
        state.completed_steps += 1
        if final:
            if normal and request.job_id is not None:
                self.policy.observe_completed_program(state.completed_steps)
            del self.programs[key]
            return
        state.pending = (request.request_id, tool, finished)
        decision = self.policy.choose_ttl(tool, request.num_tokens)
        request.continuum_pin_deadline = finished + decision.ttl
        if self.recorder is not None:
            self.recorder.record(request, ttl_decision=asdict(decision), tool=tool)

    def is_pending(self, request: Request) -> bool:
        state = self.programs.get(program_key(request))
        return (state is not None and state.pending is not None and
                state.pending[0] == request.request_id)

    def set_up_pin(self, request: Request) -> float:
        # 返回的是“完成时起算的 deadline 还剩几秒”，不是重新给一个完整 TTL。
        # pending 的 request_id 检查防止旧轮次延迟释放后，又错误地保护旧缓存。
        state = self.programs.get(program_key(request))
        # A connector's delayed free may arrive after a newer turn has started.
        if (state is None or state.pending is None or
                state.pending[0] != request.request_id):
            return 0.
        return max(0., request.continuum_pin_deadline - time.time())
