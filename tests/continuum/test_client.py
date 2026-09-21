import unittest
from unittest.mock import Mock, patch

from minisweagent.models.vllm_model import VllmModel


class ClientTests(unittest.TestCase):
    def test_default_program_id_stable_across_turns_and_unique_per_agent(self):
        with patch("minisweagent.models.vllm_model.OpenAI", return_value=Mock()):
            a, b = VllmModel(job_id=0), VllmModel(job_id=0)
            a._query([])
            first = a.client.chat.completions.create.call_args.kwargs["extra_body"]["job_id"]
            a.n_calls = 1
            a._query([])
            second = a.client.chat.completions.create.call_args.kwargs["extra_body"]["job_id"]
            self.assertEqual(first, second)
            self.assertNotEqual(a.program_id, b.program_id)

    def test_explicit_program_id_preserved(self):
        with patch("minisweagent.models.vllm_model.OpenAI", return_value=Mock()):
            self.assertEqual(VllmModel(job_id=5).program_id, "5")
