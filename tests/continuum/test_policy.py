"""Deterministic paper-equation tests; unittest, no pytest plugin dependencies."""
import importlib.util
import json
import math
import random
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

# Load the pure module directly, so these tests also run without torch/vLLM.
PATH = Path(__file__).resolve().parents[2] / "vllm/v1/core/continuum_policy.py"
spec = importlib.util.spec_from_file_location("paper_policy_under_test", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
ContinuumPolicy = module.ContinuumPolicy
ReconstructionCost = module.ReconstructionCost


class PolicyTests(unittest.TestCase):
    def policy(self, cost=3., **kwargs):
        return ContinuumPolicy(ReconstructionCost("prefill", 10000,
                                                   (0., 0., cost)), **kwargs)

    def test_cold_start_uses_exponential_prior(self):
        # arXiv v6 §4.2: Exp(1), eta=1, T initially zero; not unconditional TTL=0.
        for cost in [.25, 1., math.e, 3., 100.]:
            p = self.policy(cost)
            decision = p.choose_ttl("unseen", 100)
            expected = 0. if cost <= 1. else math.log(cost)
            self.assertAlmostEqual(decision.ttl, expected)
            self.assertEqual(decision.estimation_source, "exponential_prior")
            self.assertEqual(decision.eta, 1.)
            self.assertEqual(decision.empirical_eta, 0.)
            self.assertEqual(decision.total_tool_samples, 0)
            # Independent grid oracle for the exponential objective.
            objective = lambda t: (1. - math.exp(-t)) * cost - t
            self.assertAlmostEqual(decision.utility, objective(expected))
            for t in [i / 100 for i in range(1001)]:
                self.assertGreaterEqual(decision.utility + 1e-12, objective(t))

    def test_cold_start_uses_eta_one_and_observed_queue_cost(self):
        p = self.policy(1.)
        p.observe_completed_program(2)  # Undefined empirical eta -> 0, not prior.
        p.observe_evicted_wait(2.)
        decision = p.choose_ttl("ls", 100)
        self.assertEqual(decision.benefit, 3.)
        self.assertAlmostEqual(decision.ttl, math.log(3.))
        self.assertEqual(decision.eta, 1.)
        self.assertEqual(decision.empirical_eta, 0.)

    def test_global_threshold_is_inclusive_100(self):
        p = self.policy(3.)
        for _ in range(100):
            p.observe_tool("known", .1)
        self.assertEqual(p.choose_ttl("unseen", 100).estimation_source,
                         "exponential_prior")
        p.observe_tool("known", .1)
        decision = p.choose_ttl("unseen", 100)
        self.assertEqual(decision.estimation_source, "global_empirical")
        self.assertEqual(decision.tool_samples, 0)
        self.assertEqual(decision.cdf_samples, 101)
        self.assertEqual(decision.total_tool_samples, 101)
        self.assertEqual(decision.cold_start_threshold, 100)
        self.assertEqual(decision.ttl, .1)

    def test_per_tool_threshold_is_inclusive_100(self):
        p = self.policy(3.)
        for _ in range(101):
            p.observe_tool("fast", .1)
        for _ in range(100):
            p.observe_tool("slow", 10.)
        global_decision = p.choose_ttl("slow", 100)
        self.assertEqual(global_decision.estimation_source, "global_empirical")
        self.assertEqual(global_decision.cdf_samples, 201)
        self.assertEqual(global_decision.ttl, .1)
        p.observe_tool("slow", 10.)
        local_decision = p.choose_ttl("slow", 100)
        self.assertEqual(local_decision.estimation_source, "per_tool_empirical")
        self.assertEqual(local_decision.cdf_samples, 101)
        self.assertEqual(local_decision.total_tool_samples, 202)
        self.assertEqual(local_decision.ttl, 0.)

    def test_global_cdf_preserves_frequency_not_equal_tool_weight(self):
        p = self.policy(3.)
        for _ in range(100):
            p.observe_tool("fast", .1)
        p.observe_tool("slow", 10.)
        decision = p.choose_ttl("new", 100)
        self.assertAlmostEqual(decision.utility, 3. * 100 / 101 - .1)
        self.assertEqual(decision.ttl, .1)
        self.assertEqual(decision.cdf_samples, 101)

    def test_invalid_observations_do_not_advance_cold_start(self):
        p = self.policy()
        for _ in range(101):
            for value in [-1., float("nan"), float("inf")]:
                p.observe_tool("bad", value)
        self.assertEqual(p.choose_ttl("bad", 100).total_tool_samples, 0)
        self.assertEqual(p.choose_ttl("bad", 100).estimation_source,
                         "exponential_prior")

    def test_negative_eta_is_retained_in_both_empirical_branches(self):
        p = self.policy(3.)
        for n in [2] * 100 + [100]:
            p.observe_completed_program(n)
        self.assertLess(p.eta, 0.)
        p.observe_evicted_wait(100.)
        for _ in range(101):
            p.observe_tool("known", .5)
        for tool, source in [("known", "per_tool_empirical"),
                             ("unseen", "global_empirical")]:
            decision = p.choose_ttl(tool, 100)
            self.assertEqual(decision.estimation_source, source)
            self.assertEqual(decision.eta, p.eta)
            self.assertLess(decision.benefit, 0.)
            self.assertEqual(decision.ttl, 0.)

    def test_cdf_not_mean_threshold(self):
        p = self.policy()
        for duration in [0.1, 0.1, 10.] * 34:  # >100: per-tool empirical branch.
            p.observe_tool("ls", duration)
        self.assertEqual(p.choose_ttl("ls", 100).ttl, 0.1)

    def test_zero_utility_tie_prefers_zero(self):
        p = self.policy(1.)
        for _ in range(101):
            p.observe_tool("ls", 1.)
        self.assertEqual(p.choose_ttl("ls", 100).ttl, 0.)

    def test_random_enumeration_oracle(self):
        rng = random.Random(7)
        for _ in range(200):
            cost = rng.random() * 10
            p = self.policy(cost)
            data = [rng.choice([0., .1, .3, 1., 5.]) for _ in range(30)]
            for duration in data * 4:  # Same CDF, now past paper cold start.
                p.observe_tool("tool", duration)
            expected = max(sorted({0., *data}),
                           key=lambda t: sum(d <= t for d in data) / len(data)
                           * cost - t)
            self.assertEqual(p.choose_ttl("tool", 100).ttl, expected)

    def test_eta_is_pearson_not_regression(self):
        p = self.policy()
        for n in [2, 3, 8, 9]:
            p.observe_completed_program(n)
        pairs = [(k, n - k) for n in [2, 3, 8, 9] for k in range(1, n)]
        x, y = zip(*pairs)
        self.assertAlmostEqual(p.eta, -statistics.correlation(x, y))
        # Non-symmetric samples distinguish Pearson from a regression slope.
        pairs = [(1, 1), (2, 5), (3, 4), (4, 15)]
        x, y = zip(*pairs)
        eta = module.negative_correlation(pairs)
        self.assertAlmostEqual(eta, -statistics.correlation(x, y))
        self.assertLess(eta, 0.)
        self.assertNotAlmostEqual(eta,
                                 -statistics.covariance(x, y) / statistics.variance(x))

    def test_fixed_lengths_have_eta_one(self):
        p = self.policy()
        for _ in range(3):
            p.observe_completed_program(5)
        self.assertAlmostEqual(p.eta, 1.)

    def test_degenerate_eta_zero(self):
        p = self.policy()
        p.observe_completed_program(2)
        p.observe_completed_program(2)
        self.assertEqual(p.eta, 0.)

    def test_evicted_queue_window_enters_benefit(self):
        p = self.policy(1., queue_window=2, program_window=1)
        for wait in [100., 4., 6.]:
            p.observe_evicted_wait(wait)
        p.observe_completed_program(10)
        p.observe_completed_program(3)
        for _ in range(101):
            p.observe_tool("ls", 2.)
        decision = p.choose_ttl("ls", 100)
        self.assertEqual(decision.mean_queue_delay, 5.)
        self.assertEqual(decision.benefit, 6.)
        self.assertEqual(decision.ttl, 2.)
        self.assertEqual(list(p.completed_lengths), [3])

    def test_bad_duration_does_not_pollute_history(self):
        p = self.policy()
        for duration in [-1., float("nan"), float("inf")]:
            p.observe_tool("ls", duration)
        self.assertEqual(p.choose_ttl("ls", 100).tool_samples, 0)

    def test_quadratic_and_reload_units(self):
        self.assertAlmostEqual(ReconstructionCost(
            "prefill", 1000, (1e-6, .001, .2)).seconds(1000), 2.2)
        self.assertEqual(ReconstructionCost(
            "reload", 1000, bytes_per_token=1024,
            bytes_per_second=1024000).seconds(1000), 1.)

    def test_invalid_profile_fails_instead_of_static_fallback(self):
        with self.assertRaises(ValueError):
            ReconstructionCost("reload", 1000)
        with self.assertRaises(ValueError):
            ReconstructionCost("prefill", 1000).seconds(1001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps({"schema_version": 1, "model": "test",
                                        "mode": "prefill", "max_context": 1000,
                                        "coefficients": [0, .001, 0]}))
            with self.assertRaises(ValueError):
                ReconstructionCost.from_file(path, "wrong-model", 1000)
            with self.assertRaises(ValueError):
                ReconstructionCost.from_file(path, "test", 2000)


if __name__ == "__main__":
    unittest.main()
