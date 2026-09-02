"""CPU-only tests for the isolated MuxWise and Bullet model reproductions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.multiplex.related_work.bullet.srm import (
    BulletSRMPredictor,
    SRMCalibration,
    SRMCalibrationPoint,
    SRMHardwareProfile,
    execution_features,
)
from sglang.srt.multiplex.related_work.bullet.types import (
    ExecutionPlan,
    ExecutionState,
    KernelWork,
    LaneWork,
)
from sglang.srt.multiplex.related_work.muxwise.model import (
    ContentionGuard,
    MuxWiseProfile,
    PhaseCoefficients,
)
from sglang.srt.multiplex.related_work.muxwise.policy import MuxWiseSLOPolicy
from sglang.srt.multiplex.policy_state import (
    DecodePolicyState,
    PDMuxPolicyInput,
    PrefillPolicyState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMuxWiseModel(CustomTestCase):
    def test_paper_equations_and_runtime_max_guard(self):
        guard = ContentionGuard(
            prefill_new_boundaries=(2048,),
            prefill_reused_boundaries=(2048,),
            decode_batch_boundaries=(8,),
            decode_reused_boundaries=(2048,),
            cells=(),
            default_max_slowdown=1.2,
        )
        profile = MuxWiseProfile(
            model="llama31_8b",
            device="test",
            total_sm=108,
            prefill_models=(PhaseCoefficients("prefill", 92, (0.0, 0.0, 1.0, 1.0)),),
            decode_models=(PhaseCoefficients("decode", 16, (0.01, 1.0, 1.0)),),
            contention_guard=guard,
            metadata={},
        )
        estimate = profile.estimate(
            prefill_sm=92,
            decode_sm=16,
            prefill_new_tokens=(3, 4),
            prefill_reused_tokens=(0, 0),
            decode_context_tokens=(10, 20),
        )
        self.assertAlmostEqual(estimate.prefill_solo_ms, 8.0)
        self.assertAlmostEqual(estimate.decode_solo_ms, 3.3)
        self.assertAlmostEqual(estimate.decode_worst_case_ms, 3.96)

        refined = guard.refined(
            prefill_sm=92,
            decode_sm=16,
            prefill_new_tokens=7,
            prefill_reused_tokens=0,
            decode_batch=2,
            decode_reused_tokens=30,
            observed_slowdown=1.5,
        )
        self.assertEqual(
            refined.lookup(
                prefill_sm=92,
                decode_sm=16,
                prefill_new_tokens=7,
                prefill_reused_tokens=0,
                decode_batch=2,
                decode_reused_tokens=30,
            ),
            1.5,
        )

    def test_policy_uses_slo_then_derives_layer_quantum(self):
        guard = ContentionGuard(
            prefill_new_boundaries=(2048,),
            prefill_reused_boundaries=(2048,),
            decode_batch_boundaries=(8,),
            decode_reused_boundaries=(2048,),
            cells=(),
            default_max_slowdown=1.0,
        )
        profile = MuxWiseProfile(
            model="llama31_8b",
            device="test",
            total_sm=108,
            prefill_models=(
                PhaseCoefficients("prefill", 92, (0.0, 0.0, 0.0, 50.0)),
                PhaseCoefficients("prefill", 84, (0.0, 0.0, 0.0, 40.0)),
            ),
            decode_models=(
                PhaseCoefficients("decode", 16, (0.0, 0.0, 30.0)),
                PhaseCoefficients("decode", 24, (0.0, 0.0, 10.0)),
            ),
            contention_guard=guard,
            metadata={},
        )
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile.save(profile_path)
            policy = MuxWiseSLOPolicy(
                SimpleNamespace(
                    muxwise_profile_path=str(profile_path),
                    muxwise_tbt_slo_ms=20.0,
                    muxwise_tbt_headroom=1.0,
                    muxwise_online_guard_update=True,
                ),
                [(108, 0), (92, 16), (84, 24), (0, 108)],
            )
        policy_input = PDMuxPolicyInput(
            prefill=PrefillPolicyState((100,), (0,), 0, 32),
            decode=DecodePolicyState((100, 100)),
            current_stream_idx=1,
            trigger="test",
        )
        decision = policy.select(policy_input)
        self.assertEqual((decision.prefill_sm, decision.decode_sm), (84, 24))
        self.assertTrue(decision.tbt_satisfied)
        # ceil(10 ms * 32 full-model layers / 40 ms full prefill) = 8.
        self.assertEqual(decision.prefill_layers, 8)


class TestBulletSRM(CustomTestCase):
    def test_kernel_sum_layer_scaling_and_online_alpha(self):
        hardware = SRMHardwareProfile(
            device_name="test",
            total_sm=108,
            reference_frequency_mhz=1000,
            peak_compute_flops_s=108.0,
            peak_memory_bandwidth_Bps=108.0,
            memory_saturation_sm=54.0,
        )
        state = ExecutionState(4, 1, 2, 8, 32)
        plan = ExecutionPlan(prefill_sm=72, decode_sm=36, frequency_mhz=1000)
        features = execution_features(state, plan)
        calibration = SRMCalibration(
            points=(SRMCalibrationPoint("decode", features, 2.0),)
        )
        predictor = BulletSRMPredictor(hardware, calibration)
        decode = LaneWork((KernelWork("a", 18, 0), KernelWork("b", 18, 0)))
        prefill = LaneWork((KernelWork("p", 72, 0),))
        estimate = predictor.predict(state, decode, prefill, plan)
        # Two 0.5 s kernels sum to 1 s, then exact-state alpha=2.
        self.assertAlmostEqual(estimate.decode_latency_s, 2.0)
        self.assertAlmostEqual(estimate.prefill_latency_s, 1.0)

        refined = predictor.with_online_observation(
            phase="decode",
            state=state,
            work=decode,
            plan=plan,
            measured_latency_s=4.0,
            online_weight=0.25,
        )
        # EMA alpha: .75*2 + .25*4 = 2.5.
        self.assertAlmostEqual(
            refined.predict(state, decode, prefill, plan).decode_latency_s,
            2.5,
        )


if __name__ == "__main__":
    unittest.main()
