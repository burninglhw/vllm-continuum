import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from vllm.v1.core.estimate_with_func import ToolCallEstimator


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profile.json"
        self.path.write_text(json.dumps({
            "schema_version": 1, "mode": "prefill", "model": "local-test",
            "max_context": 256, "coefficients": [0, .001, 0],
            "dtype": "torch.float16", "tensor_parallel_size": 1,
            "enforce_eager": True}))
        self.config = NS(
            additional_config={"continuum": {"profile_path": str(self.path)}},
            model_config=NS(model="local-test", tokenizer="local-test",
                            tokenizer_mode="auto", trust_remote_code=False,
                            tokenizer_revision=None, dtype="torch.float16",
                            enforce_eager=True),
            cache_config=NS(enable_prefix_caching=True),
            scheduler_config=NS(max_model_len=256, async_scheduling=False),
            parallel_config=NS(tensor_parallel_size=1), kv_transfer_config=None)

    def test_measured_config_loads(self):
        with patch("vllm.transformers_utils.tokenizer.get_tokenizer", return_value=Mock()):
            e = ToolCallEstimator.from_config(self.config)
        self.assertAlmostEqual(e.policy.cost.seconds(100), .1)

    def test_missing_profile_is_explicit_error(self):
        self.config.additional_config = {}
        with self.assertRaisesRegex(ValueError, "profile_path"):
            ToolCallEstimator.from_config(self.config)

    def test_cache_and_async_guards(self):
        self.config.cache_config.enable_prefix_caching = False
        with self.assertRaisesRegex(ValueError, "prefix-caching"):
            ToolCallEstimator.from_config(self.config)
        self.config.cache_config.enable_prefix_caching = True
        self.config.scheduler_config.async_scheduling = True
        with self.assertRaisesRegex(ValueError, "sync scheduling"):
            ToolCallEstimator.from_config(self.config)

    def test_profile_signature_and_offload_mode_guards(self):
        self.config.parallel_config.tensor_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
            ToolCallEstimator.from_config(self.config)
        self.config.parallel_config.tensor_parallel_size = 1
        self.config.kv_transfer_config = object()
        with self.assertRaisesRegex(ValueError, "connector"):
            ToolCallEstimator.from_config(self.config)


class ProfilerTests(unittest.TestCase):
    def test_length_grid_has_three_points_and_reserves_output(self):
        from tools.continuum.profile_prefill import context_lengths
        self.assertEqual(context_lengths(8192), [1000, 2000, 4000, 8000, 8191])
        with self.assertRaises(ValueError):
            context_lengths(1000)

    def test_reads_v1_prefill_histogram(self):
        from tools.continuum.profile_prefill import prefill_snapshot
        llm = Mock()
        llm.get_metrics.return_value = [
            NS(name="vllm:request_prefill_time_seconds", count=3, sum=.9),
            NS(name="vllm:request_prefill_time_seconds", count=2, sum=.4),
            NS(name="another_metric", count=999, sum=999)]
        count, seconds = prefill_snapshot(llm)
        self.assertEqual(count, 5)
        self.assertAlmostEqual(seconds, 1.3)
