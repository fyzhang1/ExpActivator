"""Statistical integrity checks runnable with the standard-library test runner."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expactivator.analyze_state_conditioning_experiment import paired_cluster, by_id


class ClusteredPairingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"step_id": f"{u}_e_{t}", "user_id": u, "episode_id": f"{u}_e"}
                     for u in (1, 2) for t in (1, 2)]
        self.reference = {r["step_id"]: {"sr": int(r["user_id"] == 2)} for r in self.rows}

    def test_identical_predictions_have_exactly_zero_paired_interval(self):
        value = paired_cluster(self.reference, self.reference, self.rows, samples=1000)
        self.assertEqual(value["delta_ci95"], [0.0, 0.0])
        self.assertEqual((value["wins"], value["losses"]), (0, 0))

    def test_repeated_actions_are_resampled_as_whole_users(self):
        method = {key: {"sr": 1} for key in self.reference}
        value = paired_cluster(self.reference, method, self.rows, samples=2000)
        self.assertEqual(value["n_clusters"], 2)
        self.assertEqual(value["delta"], 0.5)
        self.assertEqual(value["delta_ci95"], [0.0, 1.0])
        self.assertEqual((value["wins"], value["losses"]), (2, 0))

    def test_duplicate_prediction_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            by_id([{"episode_id": "1_e_1"}, {"episode_id": "1_e_1"}])


if __name__ == "__main__":
    unittest.main()
