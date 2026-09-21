# SPDX-License-Identifier: Apache-2.0
# 中文导读：本文件只回答“某次工具调用值得保留 KV 多久”，不操作 GPU/缓存块。
# 调用方 estimate_with_func.py 提供已观测历史；scheduler.py 执行 pin/unpin。
# 阅读顺序：ReconstructionCost → negative_correlation → choose_ttl。
"""Continuum cost model: ICLR §§3.1-3.2 plus arXiv:2511.02230v6 §4.2.

All times are seconds. Only observed tool returns and completed programs
contribute statistics; no trace-provided future step counts are consumed.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


# arXiv v6 §4.2 明确给定的冷启动参数，不是经验调参或固定两秒 TTL。
COLD_START_THRESHOLD = 100
EXPONENTIAL_MEAN_SECONDS = 1.


def program_key(request):
    # job_id 标识整个多轮程序，request_id 只标识一轮请求。缺失 job_id 时不能
    # 把所有匿名请求合并成一个程序；给 request/job 加不同前缀也能避免键冲突。
    """Keep missing IDs isolated, without colliding with client job IDs."""
    if request.job_id is None:
        return ("request", request.request_id)
    return ("job", str(request.job_id))


@dataclass(frozen=True)
class ReconstructionCost:
    # 重建整段上下文的估计耗时（秒）：无 offload 用实测二次曲线；有 offload
    # 用 KV 字节数/实测带宽。这里不负责真正搬运数据，也不代表 offload 已验证。
    mode: str
    max_context: int
    coefficients: tuple[float, float, float] = (0., 0., 0.)
    bytes_per_token: float = 0.
    bytes_per_second: float = 0.

    def __post_init__(self):
        if self.max_context <= 0:
            raise ValueError("max_context must be positive")
        if self.mode == "prefill":
            if (len(self.coefficients) != 3 or
                    not all(math.isfinite(x) for x in self.coefficients)):
                raise ValueError("prefill coefficients must be finite [a,b,c]")
        elif self.mode == "reload":
            if not all(math.isfinite(x) and x > 0 for x in
                       (self.bytes_per_token, self.bytes_per_second)):
                raise ValueError("reload requires measured positive bandwidth"
                                 " and KV bytes per token")
        else:
            raise ValueError("mode must be prefill or reload")

    @classmethod
    def from_file(cls, path, model: str, max_context: int, signature=None):
        # profile 必须属于目标模型并覆盖服务的上下文上限，不能混用 8K/128K
        # 或不同模型的曲线。dtype/TP/eager 字段存在时再检查其一致性。
        with Path(path).open() as f:
            data = json.load(f)
        if data.get("schema_version") != 1:
            raise ValueError("Continuum profile schema_version must be 1")
        if data.get("model") != model:
            raise ValueError("Continuum profile model does not match server")
        for name, value in (signature or {}).items():
            if name in data and data[name] != value:
                raise ValueError(f"Continuum profile {name} does not match server")
        result = cls(
            mode=data["mode"], max_context=int(data["max_context"]),
            coefficients=tuple(data["coefficients"] if data["mode"] == "prefill"
                               else (0., 0., 0.)),
            bytes_per_token=float(data.get("bytes_per_token", 0.)),
            bytes_per_second=float(data.get("bytes_per_second", 0.)))
        if result.max_context < max_context:
            raise ValueError("Continuum profile must cover server max_model_len")
        return result

    def seconds(self, context_tokens: int) -> float:
        if not 0 <= context_tokens <= self.max_context:
            raise ValueError("context is outside calibrated profile range")
        if self.mode == "reload":
            return context_tokens * self.bytes_per_token / self.bytes_per_second
        a, b, c = self.coefficients
        return max(0., a * context_tokens**2 + b * context_tokens + c)


@dataclass(frozen=True)
class TTLDecision:
    ttl: float
    utility: float
    benefit: float
    eta: float
    mean_queue_delay: float
    reconstruction_seconds: float
    tool_samples: int
    # eta 是本次实际使用的系数；冷启动为 1，empirical_eta 保留观测值供审计。
    estimation_source: str
    cdf_samples: int
    total_tool_samples: int
    empirical_eta: float
    cold_start_threshold: int


def negative_correlation(pairs) -> float:
    # eta = -Corr(已完成步数 k, 剩余步数 N-k)。负值保留，不人为截成 0。
    # 没有足够样本或方差为零时，相关系数不可定义，本实现采用 0。
    """Paper eta, with zero for undefined correlation and no sign clipping."""
    if len(pairs) < 2:
        return 0.
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    xx = sum((x - mx)**2 for x, _ in pairs)
    yy = sum((y - my)**2 for _, y in pairs)
    if xx == 0 or yy == 0:
        return 0.
    xy = sum((x - mx) * (y - my) for x, y in pairs)
    return max(-1., min(1., -xy / math.sqrt(xx * yy)))


class ContinuumPolicy:
    def __init__(self, cost: ReconstructionCost, queue_window: int = 256,
                 program_window: int = 256):
        if queue_window < 1 or program_window < 1:
            raise ValueError("history window sizes must be positive")
        self.cost = cost
        # 工具时长按工具类型累计；排队代价/完成程序长度用有界窗口。
        # N 只有程序真正结束后才加入，不能读取 trace 的未来总步数来决策。
        self.tool_durations: dict[str, list[float]] = defaultdict(list)
        self.queue_delays: deque[float] = deque(maxlen=queue_window)
        self.completed_lengths: deque[int] = deque(maxlen=program_window)

    def observe_tool(self, tool: str, seconds: float):
        if math.isfinite(seconds) and seconds >= 0:
            self.tool_durations[tool].append(seconds)

    def observe_evicted_wait(self, seconds: float):
        if math.isfinite(seconds) and seconds >= 0:
            self.queue_delays.append(seconds)

    def observe_completed_program(self, steps: int):
        if steps > 0:
            self.completed_lengths.append(steps)

    @property
    def eta(self) -> float:
        # 一个已完成的 N 轮程序贡献 N-1 个非最终边界样本，不是每轮都已知 N。
        # Explicit sampling convention: one pair at each non-final boundary
        # of each completed program. The paper does not specify this choice.
        pairs = [(k, n - k) for n in self.completed_lengths
                 for k in range(1, n)]
        return negative_correlation(pairs)

    def choose_ttl(self, tool: str, context_tokens: int) -> TTLDecision:
        # 目标：F_tool(tau) * (eta * 平均被回收后等待 + 重建耗时) - tau。
        # v6 §4.2：|S|<=100 用 Exp(1)/eta=1；否则 |S[f]|<=100 用全局 CDF；
        # 其余用单工具 CDF。阈值处的 <= 必须保留，不能提前切换。
        tool_durations = self.tool_durations.get(tool, [])
        total_samples = sum(len(values) for values in self.tool_durations.values())
        empirical_eta = self.eta
        if total_samples <= COLD_START_THRESHOLD:
            source, durations, eta = "exponential_prior", [], 1.
        elif len(tool_durations) <= COLD_START_THRESHOLD:
            source = "global_empirical"
            # 按每次调用汇总，不对不同工具的 CDF 等权平均。
            durations = [value for values in self.tool_durations.values()
                         for value in values]
            eta = empirical_eta
        else:
            source, durations, eta = "per_tool_empirical", tool_durations, empirical_eta
        queue_delay = (sum(self.queue_delays) / len(self.queue_delays)
                       if self.queue_delays else 0.)
        reconstruction = self.cost.seconds(context_tokens)
        benefit = queue_delay * eta + reconstruction
        ttl, utility = 0., 0.
        if source == "exponential_prior":
            # 将论文 Exp(1) 代入式(2)：U(t)=(1-exp(-t/mu))*B-t，mu=1秒。
            # 最优 t = mu*log(B/mu) (B>mu)，否则为0。不是凭空指定 TTL。
            mean = EXPONENTIAL_MEAN_SECONDS
            if benefit > mean:
                ttl = mean * math.log(benefit / mean)
                utility = -math.expm1(-ttl / mean) * benefit - ttl
        else:
            # 经验 CDF 只在样本耗时处跳变，枚举 0 和全部不同样本值即可。
            counts = Counter(durations)
            cumulative = counts.get(0., 0)
            utility = cumulative / len(durations) * benefit
            for candidate in sorted(t for t in counts if t > 0):
                cumulative += counts[candidate]
                reward = cumulative / len(durations) * benefit - candidate
                # Enumeration in ascending order breaks ties toward less pinning.
                if reward > utility:
                    ttl, utility = candidate, reward
        return TTLDecision(ttl, utility, benefit, eta, queue_delay,
                           reconstruction, len(tool_durations), source,
                           len(durations), total_samples, empirical_eta,
                           COLD_START_THRESHOLD)
