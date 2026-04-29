from __future__ import annotations

import unittest

import numpy as np

from tools.paper_noise_scan import compute_boundary_artifact_metrics, iter_noise_plan


class PaperNoiseScanTest(unittest.TestCase):
    def test_boundary_metrics_use_action_jerk_at_replan_phases(self):
        actions = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        metrics = compute_boundary_artifact_metrics(actions, chunk_k=5, action_dims=6)

        self.assertEqual(metrics["action_count"], 10)
        self.assertEqual(metrics["jerk_count"], 8)
        self.assertAlmostEqual(metrics["first_boundary_transition_jerk"], 1.0)
        self.assertAlmostEqual(metrics["first_boundary_gap"], -1.0)
        self.assertAlmostEqual(metrics["boundary_interior_gap"], 0.0)

    def test_iter_noise_plan_sweeps_base_plus_direction(self):
        base = np.ones((2, 3), dtype=np.float32)
        direction = np.full((2, 3), 2.0, dtype=np.float32)

        plan = list(iter_noise_plan(base_noise=base, direction=direction, alphas=[-0.5, 0.0, 0.5]))

        self.assertEqual([item["noise_id"] for item in plan], ["alpha_-0.5", "alpha_0", "alpha_0.5"])
        np.testing.assert_allclose(plan[0]["noise"], np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_allclose(plan[1]["noise"], base)
        np.testing.assert_allclose(plan[2]["noise"], np.full((2, 3), 2.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
