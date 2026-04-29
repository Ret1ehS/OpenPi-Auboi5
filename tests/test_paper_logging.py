from __future__ import annotations

import types
import unittest

import numpy as np

import main


class PaperLoggingPayloadTest(unittest.TestCase):
    def test_trial_id_increments_trailing_number_with_padding(self):
        self.assertEqual(main.next_paper_trial_id("trial_001", 0), "trial_001")
        self.assertEqual(main.next_paper_trial_id("trial_001", 1), "trial_002")
        self.assertEqual(main.next_paper_trial_id("baseline-099", 2), "baseline-101")

    def test_trial_id_appends_counter_when_no_trailing_number(self):
        self.assertEqual(main.next_paper_trial_id("baseline", 0), "baseline_0001")
        self.assertEqual(main.next_paper_trial_id("baseline", 1), "baseline_0002")

    def test_action_chunk_payload_contains_reconstructable_actions_and_context(self):
        raw_actions = np.arange(21, dtype=np.float32).reshape(3, 7)
        exec_actions = raw_actions[:2]
        tcp_deltas = exec_actions[:, :6]
        noise = np.ones((3, 7), dtype=np.float32)

        payload = main.build_paper_action_chunk_log(
            session_id="sess-1",
            trial_id="trial-7",
            condition="target-low",
            step=4,
            prompt="open the drawer",
            chunk_k=5,
            raw_actions=raw_actions,
            exec_actions=exec_actions,
            tcp_deltas=tcp_deltas,
            start_pose_sim=np.arange(6, dtype=np.float32),
            observation_state=np.arange(7, dtype=np.float32),
            observation_tick=12,
            submit_observation_tick=13,
            noise_seed=123,
            noise_id="z03",
            noise=noise,
        )

        self.assertEqual(payload["event"], "action_chunk")
        self.assertEqual(payload["schema"], "paper_action_log_v1")
        self.assertEqual(payload["session_id"], "sess-1")
        self.assertEqual(payload["trial_id"], "trial-7")
        self.assertEqual(payload["condition"], "target-low")
        self.assertEqual(payload["step"], 4)
        self.assertEqual(payload["replan_idx"], 3)
        self.assertEqual(payload["chunk_k"], 5)
        self.assertEqual(payload["raw_actions"], raw_actions.tolist())
        self.assertEqual(payload["exec_actions"], exec_actions.tolist())
        self.assertEqual(payload["tcp_deltas"], tcp_deltas.tolist())
        self.assertEqual(payload["start_pose_sim"], np.arange(6, dtype=np.float32).tolist())
        self.assertEqual(payload["observation_state"], np.arange(7, dtype=np.float32).tolist())
        self.assertEqual(payload["observation_tick"], 12)
        self.assertEqual(payload["submit_observation_tick"], 13)
        self.assertEqual(payload["noise_seed"], 123)
        self.assertEqual(payload["noise_id"], "z03")
        self.assertEqual(payload["noise_shape"], [3, 7])
        self.assertEqual(payload["noise"], noise.tolist())

    def test_submitted_plan_payload_serializes_every_retimed_step(self):
        steps = [
            types.SimpleNamespace(
                pose_sim=np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
                pose_real=np.array([6, 5, 4, 3, 2, 1], dtype=np.float32),
                source_action_index=0,
            ),
            types.SimpleNamespace(
                pose_sim=np.array([2, 3, 4, 5, 6, 7], dtype=np.float32),
                pose_real=np.array([7, 6, 5, 4, 3, 2], dtype=np.float32),
                source_action_index=1,
            ),
        ]

        payload = main.build_paper_submitted_plan_log(
            session_id="sess-1",
            trial_id="trial-7",
            condition="baseline",
            step=2,
            prompt="open the drawer",
            observation_tick=5,
            submit_observation_tick=6,
            steps=steps,
        )

        self.assertEqual(payload["event"], "submitted_plan")
        self.assertEqual(payload["schema"], "paper_action_log_v1")
        self.assertEqual(payload["replan_idx"], 1)
        self.assertEqual(payload["submitted_plan"][0]["source_action_index"], 0)
        self.assertEqual(payload["submitted_plan"][0]["pose_sim"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(payload["submitted_plan"][0]["pose_real"], [6, 5, 4, 3, 2, 1])
        self.assertEqual(payload["submitted_plan"][1]["source_action_index"], 1)

    def test_session_outcome_payload_records_manual_success_label(self):
        payload = main.build_paper_session_outcome_log(
            session_id="sess-1",
            trial_id="trial-7",
            condition="baseline",
            prompt="open the drawer",
            steps=12,
            success=False,
            outcome_source="manual",
            failure_reason="missed handle",
        )

        self.assertEqual(payload["event"], "session_outcome")
        self.assertFalse(payload["success"])
        self.assertEqual(payload["outcome_source"], "manual")
        self.assertEqual(payload["failure_reason"], "missed handle")


if __name__ == "__main__":
    unittest.main()
