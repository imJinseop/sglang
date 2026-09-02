"""MI cost-model policy for choosing among pre-created PDMux partitions."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.multiplex.policy_state import PDMuxPolicyInput

if TYPE_CHECKING:
    from sglang.srt.multiplex.adaptive_pdmux_config import AdaptivePDMuxConfig


@dataclass(frozen=True)
class MIPDMuxDecision:
    stream_idx: int
    prefill_sm: int
    decode_sm: int
    frequency_mhz: int
    decode_latency_ms: float
    prefill_latency_ms: float
    blended_latency_ms: float
    energy_j: float
    power_w: float
    sustainable: bool
    throttle_factor: float
    effective_frequency_mhz: float
    candidates_evaluated: int
    decision_time_ms: float
    prefill_tokens: int
    decode_batch: int
    decode_context_tokens: int
    remaining_layers: int
    prefill_layers: int
    decode_layers: int


class MIPDMuxPolicy:
    """Run MI online, but return only an already-created PDMux stream group.

    PDMux cannot resize a Green Context in place.  The policy therefore gives
    MI exactly the finite set of internal ``(prefill_sm, decode_sm)`` groups
    whose decode CUDA graphs were captured during engine initialization.
    """

    def __init__(
        self,
        config: AdaptivePDMuxConfig,
        sm_counts: list[tuple[int, int]],
    ) -> None:
        if config.mi_path and config.mi_path not in sys.path:
            sys.path.insert(0, config.mi_path)

        try:
            from bullet_green import (
                BulletGreenPlanner,
                ExecutionPlan,
                ExecutionState,
                MICostModelPredictor,
                Objective,
                ScheduleConstraints,
                build_llm_layer_work,
            )
        except ImportError as exc:
            location = config.mi_path or "the current Python path"
            raise ImportError(
                "partition_policy='mi_cost_model' requires the MI repository; "
                f"could not import bullet_green from {location}"
            ) from exc

        predictor = MICostModelPredictor.from_device_name(
            config.mi_device_name,
            device_dir=config.mi_device_dir,
            dtype=config.mi_dtype,
        )
        if predictor.total_sm != sum(sm_counts[1]):
            raise ValueError(
                "MI device profile and PDMux GPU disagree on total SM count: "
                f"MI={predictor.total_sm}, PDMux={sum(sm_counts[1])}"
            )

        frequency_mhz = config.mi_frequency_mhz or max(
            predictor.frequencies_mhz
        )
        if frequency_mhz not in predictor.frequencies_mhz:
            raise ValueError(
                f"mi_frequency_mhz={frequency_mhz} is not calibrated; "
                f"available frequencies are {predictor.frequencies_mhz}"
            )

        self._ExecutionState = ExecutionState
        self._build_llm_layer_work = build_llm_layer_work
        self._model = config.mi_workload_model
        self._dtype = config.mi_dtype
        self._frequency_mhz = frequency_mhz
        self._split_forward_token_budget = config.split_forward_token_budget
        self._constraints = ScheduleConstraints(
            allow_modeled_throttling=config.mi_allow_modeled_throttling
        )
        self._planner = BulletGreenPlanner(
            predictor,
            total_sm=predictor.total_sm,
            frequencies_mhz=(frequency_mhz,),
            sm_granularity=2,
            objective=Objective(config.mi_objective),
        )

        self._plans = tuple(
            ExecutionPlan(
                decode_sm=decode_sm,
                prefill_sm=prefill_sm,
                frequency_mhz=frequency_mhz,
            )
            for prefill_sm, decode_sm in sm_counts[1:-1]
        )
        self._stream_idx = {
            (plan.decode_sm, plan.prefill_sm): stream_idx
            for stream_idx, plan in enumerate(self._plans, start=1)
        }
        if not self._plans:
            raise ValueError("MI PDMux policy requires at least one paired partition")

    @property
    def frequency_mhz(self) -> int:
        return self._frequency_mhz

    @property
    def candidate_count(self) -> int:
        return len(self._plans)

    def select(self, policy_input: PDMuxPolicyInput) -> MIPDMuxDecision:
        start = time.perf_counter()
        prefill = policy_input.prefill
        decode = policy_input.decode
        total_layers = prefill.total_layers
        remaining_layers = prefill.remaining_layers
        prefill_layers = min(
            remaining_layers,
            max(
                1,
                self._split_forward_token_budget
                // max(1, prefill.new_tokens),
            ),
        )
        state = self._ExecutionState(
            prefill_tokens=prefill.new_tokens,
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
        decode_work = layer_work.decode.scaled(total_layers)
        prefill_work = (
            layer_work.prefill
            if prefill_layers == 1
            else layer_work.prefill.scaled(prefill_layers)
        )
        decision = self._planner.select(
            state,
            decode_work,
            prefill_work,
            constraints=self._constraints,
            plans=self._plans,
        )
        estimate = decision.estimate
        stream_idx = self._stream_idx[
            (decision.plan.decode_sm, decision.plan.prefill_sm)
        ]
        return MIPDMuxDecision(
            stream_idx=stream_idx,
            prefill_sm=decision.plan.prefill_sm,
            decode_sm=decision.plan.decode_sm,
            frequency_mhz=decision.plan.frequency_mhz,
            decode_latency_ms=estimate.decode_latency_s * 1e3,
            prefill_latency_ms=estimate.prefill_latency_s * 1e3,
            blended_latency_ms=estimate.blended_latency_s * 1e3,
            energy_j=estimate.total_energy_j,
            power_w=estimate.power_demand_w,
            sustainable=estimate.sustainable,
            throttle_factor=estimate.throttle_factor,
            effective_frequency_mhz=(
                decision.plan.frequency_mhz / estimate.throttle_factor
            ),
            candidates_evaluated=decision.candidates_evaluated,
            decision_time_ms=(time.perf_counter() - start) * 1e3,
            prefill_tokens=state.prefill_tokens,
            decode_batch=state.decode_batch,
            decode_context_tokens=state.decode_context_tokens,
            remaining_layers=state.remaining_layers,
            prefill_layers=prefill_layers,
            decode_layers=total_layers,
        )
