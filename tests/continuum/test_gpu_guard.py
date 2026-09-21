import unittest
import signal
from unittest.mock import Mock, patch
from tools.continuum.guard_gpu import stop_owned_group, violations


class GuardTests(unittest.TestCase):
    def sample(self, processes=None, free=120000, activity=None):
        return {"processes": processes or {10: 1900}, "free_mib": free,
                "activity": activity or {}}

    def check(self, sample, own={20}):
        return violations(sample, self.sample(), own, 28000, 100000, 512)

    def test_own_process_does_not_trigger_foreign_activity(self):
        reasons, memory = self.check(self.sample({10: 1900, 20: 23000}, activity={20: {"sm": 100}}))
        self.assertEqual(reasons, [])
        self.assertEqual(memory, 23000)

    def test_foreign_growth_and_new_process(self):
        reasons, _ = self.check(self.sample({10: 2500, 30: 1}))
        self.assertIn("new_foreign_gpu_process", reasons)
        self.assertIn("foreign_memory_growth", reasons)

    def test_foreign_compute(self):
        reasons, _ = self.check(self.sample(activity={10: {"sm": 1}}))
        self.assertIn("foreign_compute_activity", reasons)

    def test_unknown_activity_is_not_proof_of_idle(self):
        reasons, _ = self.check(self.sample(activity={10: {"sm": None, "memory": None}}))
        self.assertEqual(reasons, [])

    def test_memory_limits(self):
        reasons, _ = self.check(self.sample({10: 1900, 20: 29000}, free=90000))
        self.assertIn("own_memory_cap_exceeded", reasons)
        self.assertIn("free_memory_below_floor", reasons)

    def test_never_signals_an_unexpected_process_group(self):
        child = Mock(pid=200)
        child.poll.return_value = None
        with patch("os.getpgid", return_value=300), patch("os.killpg") as kill:
            with self.assertRaises(RuntimeError):
                stop_owned_group(child)
            kill.assert_not_called()

    def test_only_signals_own_live_group(self):
        child = Mock(pid=200)
        child.poll.return_value = None
        with patch("os.getpgid", return_value=200), patch("os.killpg") as kill:
            stop_owned_group(child)
            kill.assert_called_once_with(200, signal.SIGTERM)

    def test_finished_child_never_signaled(self):
        child = Mock(pid=200)
        child.poll.return_value = 0
        with patch("os.killpg") as kill:
            stop_owned_group(child)
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
