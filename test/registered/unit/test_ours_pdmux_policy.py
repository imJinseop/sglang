"""CPU-only contract tests for the external energy-aware policy port."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.multiplex.ours_pdmux_policy import OursPDMuxPolicy
from sglang.srt.multiplex.policy_state import (
    DecodePolicyState,
    PDMuxPolicyInput,
    PrefillPolicyState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config():
    return SimpleNamespace(
        ours_objective="energy",
        ours_tpot_slo_ms=12.5,
        ours_policy_entrypoint="test_policy:factory",
        split_forward_token_budget=4096,
    )


def _policy_input():
    return PDMuxPolicyInput(
        prefill=PrefillPolicyState(
            new_tokens_by_request=(2048,),
            reused_tokens_by_request=(0,),
            current_layer=4,
            total_layers=32,
        ),
        decode=DecodePolicyState(context_tokens_by_request=(3, 3)),
        current_stream_idx=1,
        trigger="test",
    )


class _FixedPolicy:
    def __init__(self, context, stream_idx):
        self.context = context
        self.stream_idx = stream_idx

    def select(self, state):
        return {
            "stream_idx": self.stream_idx,
            "reason": "test",
            "predicted_latency_ms": 7.0,
            "predicted_energy_j": 0.5,
            "predicted_power_w": 200.0,
            "slo_satisfied": True,
        }


class TestOursPDMuxPolicy(CustomTestCase):
    def test_context_exposes_only_precreated_paired_candidates(self):
        with patch(
            "sglang.srt.multiplex.ours_pdmux_policy._load_factory",
            return_value=lambda context: _FixedPolicy(context, 2),
        ):
            policy = OursPDMuxPolicy(
                _config(), [(108, 0), (84, 24), (68, 40), (0, 108)]
            )

        self.assertEqual(policy.context.objective, "energy")
        self.assertEqual(policy.context.tpot_slo_ms, 12.5)
        self.assertEqual(
            [candidate.stream_idx for candidate in policy.context.candidates],
            [1, 2],
        )
        decision = policy.select(_policy_input())
        self.assertEqual(decision.selection.stream_idx, 2)
        self.assertEqual(decision.selection.predicted_energy_j, 0.5)
        self.assertEqual(decision.state.prefill_layers_in_quantum, 2)
        self.assertEqual(decision.state.decode_context_tokens, 6)

    def test_runtime_rejects_non_candidate_stream_index(self):
        with patch(
            "sglang.srt.multiplex.ours_pdmux_policy._load_factory",
            return_value=lambda context: _FixedPolicy(context, 99),
        ):
            policy = OursPDMuxPolicy(
                _config(), [(108, 0), (84, 24), (68, 40), (0, 108)]
            )

        with self.assertRaisesRegex(ValueError, "fixed paired candidates"):
            policy.select(_policy_input())


if __name__ == "__main__":
    unittest.main()
