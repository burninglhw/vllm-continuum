# SPDX-License-Identifier: Apache-2.0
# 中文导读：这里只负责排队次序，不计算 TTL、不分配显存。
# v6 §4.3：被抢占请求优先 → 其余请求按 pin 分组 → 组内按程序级 FCFS。
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.continuum_policy import program_key

class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    CONTINUUM = "continuum"

class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))

class ContinuumRequestQueue(deque[Request], RequestQueue):
    """arXiv v6 §4.3: preempted, then pinned, then program-level FCFS."""

    def __init__(self) -> None:
        super().__init__()
        self.job_id_first_entry_time = {}

    def _remember(self, request):
        # 同一多轮程序记住最初到达时间，不能每轮返回都重新排到队尾。
        self.job_id_first_entry_time.setdefault(program_key(request),
                                                request.arrival_time)

    def forget_program(self, request):
        # 程序结束后清理；以后即便客户端复用 ID，也不应继承旧程序的优先级。
        self.job_id_first_entry_time.pop(program_key(request), None)

    def add_request(self, request: Request) -> None:
        self._remember(request)
        self.append(request)

    def peek_request(self, pinned_requests=(), kv_cache_manager=None,
                     connector=None) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        pinned = {program_key(req) for req, _ in pinned_requests}
        # 先恢复因 running 争用而抢占的请求。TTL 仅区分非抢占请求；每组内
        # 按程序最初到达排序。request.arrival_time 只是相同程序时间的平局规则。
        return min(self, key=lambda req: (
            0 if req.status == RequestStatus.PREEMPTED else
            1 if program_key(req) in pinned else 2,
            self.job_id_first_entry_time.get(program_key(req), req.arrival_time),
            req.arrival_time))

    def pop_request(self, pinned_requests=(), kv_cache_manager=None,
                    connector=None) -> Request:
        request = self.peek_request(pinned_requests)
        self.remove(request)
        return request

    def prepend_request(self, request: Request) -> None:
        self._remember(request)
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self._remember(request)
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        removed = set(requests)
        remaining = [request for request in self if request not in removed]
        self.clear()
        self.extend(remaining)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        return super().__reversed__()

def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.CONTINUUM:
        return ContinuumRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
