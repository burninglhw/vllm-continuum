"""Exact-token replay AFTER a real model forward, for scheduler experiments.

This does not evaluate generation quality. Both scheduling policies must use
the same processor and token sequences. No trace timing reaches the scheduler.
"""
import torch

from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class ReplayTokens(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}
        self.device = device

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        def state(params, prompt_ids, output_ids):
            expected = (params.extra_args or {}).get("replay_token_ids")
            return (expected, output_ids) if expected is not None else None
        process_dict_updates(self.requests, batch_update, state)

    def apply(self, logits):
        rows, tokens = [], []
        for row, (expected, generated) in self.requests.items():
            if len(generated) >= len(expected):
                raise RuntimeError("Replay continued past its terminal token")
            rows.append(row)
            tokens.append(expected[len(generated)])
        if rows:
            indices = torch.tensor(rows, device=self.device, dtype=torch.long)
            token_ids = torch.tensor(tokens, device=self.device, dtype=torch.long)
            logits[indices] = -float("inf")
            logits[indices, token_ids] = 0
        return logits
