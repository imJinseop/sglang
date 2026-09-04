"""Bullet SRM and TPOT-first dispatch over fixed PDMux candidates.

The paper-derived SRM, workload lifting, and policy live together in this
directory. MI is used only to describe per-kernel Llama work and measured GPU
capacity; it is not used to predict Bullet-mode latency or energy.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.multiplex.policy_state import PDMuxPolicyInput

if TYPE_CHECKING:
    from sglang.srt.multiplex.adaptive_pdmux_config import AdaptivePDMuxConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BulletPDMuxDecision:
    stream_idx: int
    prefill_sm: int
    decode_sm: int
    frequency_mhz: int
    predicted_tpot_ms: float
    predicted_prefill_quantum_ms: float
    predicted_remaining_prefill_ms: float
    tpot_limit_ms: float
    tpot_satisfied: bool
    reason: str
    candidates_evaluated: int
    feasible_candidates: int
    decision_time_ms: float
    prefill_tokens: int
    prefill_batch: int
    decode_batch: int
    decode_context_tokens: int
    remaining_layers: int
    prefill_layers: int
    decode_layers: int


class BulletSRMTPOTPolicy:
    """Paper-style SRM plus prefill-priority TPOT provisioning.

    Candidate plans are restricted to ``sm_counts[1:-1]``.  The policy never
    constructs a Green Context at runtime.  Bullet's paper uses a fixed number
    of prefill layers per control step; ``bullet_prefill_layers_per_step`` is
    that control quantum in this common PDMux substrate.
    """

    def __init__(
        self,
        config: AdaptivePDMuxConfig,
        sm_counts: list[tuple[int, int]],
        *,
        model_config,
    ) -> None:
        if config.mi_path and config.mi_path not in sys.path:
            sys.path.insert(0, config.mi_path)

        try:
            from .srm import BulletSRMPredictor, SRMHardwareProfile
            from .types import ExecutionPlan, ExecutionState
            from .workloads import build_llm_layer_work
        except ImportError as exc:
            location = config.mi_path or "the current Python path"
            raise ImportError(
                "partition_policy='bullet_srm_tpot' requires MI's device and "
                f"workload modules; could not import them from {location}"
            ) from exc

        if config.bullet_srm_profile_path:
            predictor = BulletSRMPredictor.load(config.bullet_srm_profile_path)
        else:
            hardware = SRMHardwareProfile.from_device_name(
                config.bullet_device_name,
                dtype=config.bullet_dtype,
                reference_frequency_mhz=config.bullet_reference_frequency_mhz,
                device_dir=config.bullet_device_dir,
            )
            predictor = BulletSRMPredictor(hardware)

        total_sm = sum(sm_counts[1])
        if predictor.total_sm != total_sm:
            raise ValueError(
                "Bullet SRM profile and PDMux GPU disagree on total SM count: "
                f"SRM={predictor.total_sm}, PDMux={total_sm}"
            )
        if predictor.reference_frequency_mhz != config.bullet_reference_frequency_mhz:
            raise ValueError(
                "loaded Bullet SRM reference frequency does not match config: "
                f"profile={predictor.reference_frequency_mhz}, "
                f"config={config.bullet_reference_frequency_mhz}"
            )

        self._ExecutionState = ExecutionState
        self._build_llm_layer_work = build_llm_layer_work
        self._predictor = predictor
        self._model = model_config
        self._dtype = config.bullet_dtype
        self._frequency_mhz = config.bullet_reference_frequency_mhz
        self._tpot_slo_ms = config.bullet_tpot_slo_ms
        self._tpot_headroom = config.bullet_tpot_headroom
        self._prefill_layers_per_step = config.bullet_prefill_layers_per_step
        self._online_calibration = config.bullet_online_calibration
        self._online_calibration_weight = config.bullet_online_calibration_weight

        if not predictor.calibrated:
            logger.warning(
                "Bullet SRM has no sparse alpha calibration points. This is an "
                "uncalibrated roofline smoke mode, not a paper-quality Bullet "
                "comparison. Supply bullet_srm_profile_path before evaluation."
            )

        self._plans = tuple(
            ExecutionPlan(
                decode_sm=decode_sm,
                prefill_sm=prefill_sm,
                frequency_mhz=self._frequency_mhz,
            )
            for prefill_sm, decode_sm in sm_counts[1:-1]
        )
        if not self._plans:
            raise ValueError("Bullet PDMux policy requires a paired partition")
        self._stream_idx = {
            (plan.decode_sm, plan.prefill_sm): stream_idx
            for stream_idx, plan in enumerate(self._plans, start=1)
        }

    @property
    def frequency_mhz(self) -> int:
        return self._frequency_mhz

    @property
    def candidate_count(self) -> int:
        return len(self._plans)

    @property
    def effective_tpot_ms(self) -> float:
        return self._tpot_slo_ms * self._tpot_headroom

    def select(self, policy_input: PDMuxPolicyInput) -> BulletPDMuxDecision:
        start = time.perf_counter()
        prefill = policy_input.prefill
        decode = policy_input.decode
        prefill_tokens = prefill.new_tokens
        total_layers = prefill.total_layers
        remaining_layers = prefill.remaining_layers
        prefill_layers = min(remaining_layers, self._prefill_layers_per_step)
        state = self._ExecutionState(
            prefill_tokens=prefill_tokens,
            prefill_batch=prefill.batch_size,
            decode_batch=decode.batch_size,
            decode_context_tokens=decode.context_tokens,
            remaining_layers=remaining_layers,
        )
        layer_work = self._build_llm_layer_work(
            state,
            model=self._model,
            dtype=self._dtype,
        )
        # One decode iteration always traverses the complete model.  Prefill is
        # the fixed layer quantum that Bullet will launch before asking again.
        decode_work = layer_work.decode.scaled(total_layers)
        prefill_work = layer_work.prefill.scaled(prefill_layers)
        estimates = tuple(
            self._predictor.predict(state, decode_work, prefill_work, plan)
            for plan in self._plans
        )
        feasible = tuple(
            estimate
            for estimate in estimates
            if estimate.decode_latency_s * 1e3 <= self.effective_tpot_ms
        )
        if feasible:
            selected = min(
                feasible,
                key=lambda estimate: (
                    estimate.plan.decode_sm,
                    estimate.prefill_latency_s,
                    estimate.decode_latency_s,
                ),
            )
            reason = "minimum_decode_sm_satisfying_tpot"
            tpot_satisfied = True
        else:
            selected = min(
                estimates,
                key=lambda estimate: (
                    estimate.decode_latency_s,
                    -estimate.plan.decode_sm,
                ),
            )
            reason = "tpot_overload_best_decode_candidate"
            tpot_satisfied = False

        stream_idx = self._stream_idx[
            (selected.plan.decode_sm, selected.plan.prefill_sm)
        ]
        per_layer_prefill_s = selected.prefill_latency_s / prefill_layers
        return BulletPDMuxDecision(
            stream_idx=stream_idx,
            prefill_sm=selected.plan.prefill_sm,
            decode_sm=selected.plan.decode_sm,
            frequency_mhz=selected.plan.frequency_mhz,
            predicted_tpot_ms=selected.decode_latency_s * 1e3,
            predicted_prefill_quantum_ms=selected.prefill_latency_s * 1e3,
            predicted_remaining_prefill_ms=(
                per_layer_prefill_s * remaining_layers * 1e3
            ),
            tpot_limit_ms=self.effective_tpot_ms,
            tpot_satisfied=tpot_satisfied,
            reason=reason,
            candidates_evaluated=len(estimates),
            feasible_candidates=len(feasible),
            decision_time_ms=(time.perf_counter() - start) * 1e3,
            prefill_tokens=state.prefill_tokens,
            prefill_batch=state.prefill_batch,
            decode_batch=state.decode_batch,
            decode_context_tokens=state.decode_context_tokens,
            remaining_layers=state.remaining_layers,
            prefill_layers=prefill_layers,
            decode_layers=total_layers,
        )

    def observe_phase(
        self,
        decision: BulletPDMuxDecision,
        *,
        phase: str,
        observed_latency_ms: float,
    ) -> bool:
        """Refine one exact execution-state alpha from a runtime observation."""

        if not self._online_calibration:
            return False
        state = self._ExecutionState(
            prefill_tokens=decision.prefill_tokens,
            prefill_batch=decision.prefill_batch,
            decode_batch=decision.decode_batch,
            decode_context_tokens=decision.decode_context_tokens,
            remaining_layers=decision.remaining_layers,
        )
        work = self._build_llm_layer_work(
            state,
            model=self._model,
            dtype=self._dtype,
        )
        plan = next(
            candidate
            for candidate in self._plans
            if candidate.prefill_sm == decision.prefill_sm
            and candidate.decode_sm == decision.decode_sm
        )
        if phase == "decode":
            lane_work = work.decode.scaled(decision.decode_layers)
        elif phase == "prefill":
            lane_work = work.prefill.scaled(decision.prefill_layers)
        else:
            raise ValueError("phase must be 'decode' or 'prefill'")
        self._predictor = self._predictor.with_online_observation(
            phase=phase,
            state=state,
            work=lane_work,
            plan=plan,
            measured_latency_s=observed_latency_ms / 1e3,
            online_weight=self._online_calibration_weight,
        )
        return True
