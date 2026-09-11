# MI project extension; not part of upstream SGLang.
"""Ours: minimum decode SM meeting TPOT at a predicted power-feasible clock."""

from __future__ import annotations

import sys
import time
import logging
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from sglang.srt.mi_ext.multiplex.policy_state import PDMuxPolicyInput

if TYPE_CHECKING:
    from sglang.srt.mi_ext.multiplex.adaptive_pdmux_config import AdaptivePDMuxConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MIPDMuxDecision:
    stream_idx: int
    prefill_sm: int
    decode_sm: int
    frequency_mhz: int
    decode_latency_ms: float
    prefill_latency_ms: float
    power_w: float
    power_limit_w: float
    power_limit_feasible: bool
    effective_frequency_mhz: int
    power_at_requested_frequency_w: float
    frequency_evaluations: int
    frequency_search: str
    candidate_predictions: tuple[dict, ...]
    candidates_evaluated: int
    feasible_candidates: int
    tpot_limit_ms: float
    tpot_satisfied: bool
    reason: str
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
    MI exactly the finite set of internal ``(prefill_sm, decode_sm)`` groups.
    Partitioned execution captures a decode graph set per group; SM oversubscription
    shares its full-SM decode graph set across these prefill choices.
    """

    def __init__(
        self,
        config: AdaptivePDMuxConfig,
        sm_counts: list[tuple[int, int]],
        *,
        model_config,
        page_size: int,
    ) -> None:
        if config.mi_path and config.mi_path not in sys.path:
            sys.path.insert(0, config.mi_path)

        try:
            from bullet_green import (
                ExecutionPlan,
                ExecutionState,
                MISGLangRuntimePredictor,
            )
            from modeling.cost_model.workload import get_model_spec
        except ImportError as exc:
            location = config.mi_path or "the current Python path"
            raise ImportError(
                "partition_policy='mi' requires the MI repository; "
                f"could not import bullet_green from {location}"
            ) from exc

        predictor = MISGLangRuntimePredictor.from_device_name(
            config.mi_device_name,
            model_config=model_config,
            page_size=page_size,
            device_dir=config.mi_device_dir,
            dtype=config.mi_dtype,
            allow_extrapolation=config.mi_allow_extrapolation,
            calibration_path=config.mi_calibration_path,
            power_limit_w=config.mi_power_limit_w,
        )
        logger.info(
            "Ours selects minimum decode SM satisfying TPOT at a shared power-feasible clock"
        )
        if predictor.total_sm != sum(sm_counts[1]):
            raise ValueError(
                "MI device profile and PDMux GPU disagree on total SM count: "
                f"MI={predictor.total_sm}, PDMux={sum(sm_counts[1])}"
            )

        frequency_mhz = config.mi_frequency_mhz or max(predictor.frequencies_mhz)
        if frequency_mhz not in predictor.frequencies_mhz:
            raise ValueError(
                f"mi_frequency_mhz={frequency_mhz} is not a supported GPU clock; "
                f"available frequencies are {predictor.frequencies_mhz}"
            )

        self._ExecutionState = ExecutionState
        # The predictor expands model invocations itself. No legacy roofline
        # workload construction or Bullet SRM is involved in this policy.
        # Model dimensions come from the exact Hugging Face config SGLang has
        # already loaded for ``model_path``. Device terms remain exclusively in
        # device/<device>/calibration.json and the device's dispatch tables.
        self._model = get_model_spec(model_config)
        self._frequency_mhz = frequency_mhz
        self._split_forward_token_budget = config.split_forward_token_budget
        self._tpot_slo_ms = config.mi_tpot_slo_ms
        self._predictor = predictor

        self._plans = tuple(
            ExecutionPlan(
                decode_sm=decode_sm,
                prefill_sm=prefill_sm,
                frequency_mhz=frequency_mhz,
            )
            for prefill_sm, decode_sm in sorted(sm_counts[1:-1], key=lambda pair: pair[1])
        )
        self._stream_idx = {
            (d, p): i for i, (p, d) in enumerate(sm_counts[1:-1], start=1)
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
        if total_layers != self._model.num_layers:
            raise ValueError("scheduler layer count differs from the MI model config")
        if prefill.reused_tokens:
            raise NotImplementedError(
                "Ours step predictor currently supports fresh-prompt prefill; "
                "prefix-cache attention is not modeled"
            )
        remaining_layers = prefill.remaining_layers
        prefill_layers = min(
            remaining_layers,
            max(
                1,
                self._split_forward_token_budget // max(1, prefill.new_tokens),
            ),
        )
        state = self._ExecutionState(
            prefill_tokens=prefill.new_tokens,
            prefill_batch=prefill.batch_size,
            decode_batch=decode.batch_size,
            decode_context_tokens=decode.context_tokens,
            remaining_layers=remaining_layers,
            prefill_layers_in_quantum=prefill_layers,
            decode_layers_in_quantum=total_layers,
        )
        estimates = []
        for plan in self._plans:
            try:
                current = self._predictor.predict_execution(state, plan)
                estimates.append(current)
                if current.power_limit_feasible and current.decode_latency_s*1e3 <= self._tpot_slo_ms:
                    break  # Sorted SM menu: larger decode allocations cannot win.
            except NotImplementedError as exc:
                # Preserve the actual ragged state as well as the projected
                # predictor dimensions. This candidate has not been selected
                # or submitted to the GPU; do not report a kernel launch here.
                context = {
                    "policy_input": asdict(policy_input),
                    "execution_state": asdict(state),
                    "candidate_plan": asdict(plan),
                    "candidate_submitted": False,
                }
                raise NotImplementedError(
                    f"{exc}\nMI_POLICY_INPUT_JSON="
                    + json.dumps(context, separators=(",", ":"))
                ) from exc
        power_feasible = [x for x in estimates if x.power_limit_feasible]
        feasible = [x for x in power_feasible if x.decode_latency_s*1e3 <= self._tpot_slo_ms]
        if feasible:
            estimate = min(feasible, key=lambda x: x.plan.decode_sm)
            reason = "minimum_decode_sm_within_tpot_slo"
        else:
            estimate = min(power_feasible or estimates,
                           key=lambda x: (x.decode_latency_s, x.plan.decode_sm))
            reason = ("no_feasible_candidate_minimum_decode_latency" if power_feasible
                      else "no_power_feasible_candidate")
        stream_idx = self._stream_idx[
            (estimate.plan.decode_sm, estimate.plan.prefill_sm)
        ]
        return MIPDMuxDecision(
            stream_idx=stream_idx,
            prefill_sm=estimate.plan.prefill_sm,
            decode_sm=estimate.plan.decode_sm,
            frequency_mhz=estimate.plan.frequency_mhz,
            decode_latency_ms=estimate.decode_latency_s * 1e3,
            prefill_latency_ms=estimate.prefill_latency_s * 1e3,
            power_w=estimate.power_w,
            power_limit_w=estimate.power_limit_w,
            power_limit_feasible=estimate.power_limit_feasible,
            effective_frequency_mhz=estimate.effective_frequency_mhz,
            power_at_requested_frequency_w=estimate.power_at_requested_frequency_w,
            frequency_evaluations=sum(x.frequency_evaluations for x in estimates),
            frequency_search=estimate.frequency_search,
            candidate_predictions=tuple(asdict(x) for x in estimates),
            candidates_evaluated=len(estimates),
            feasible_candidates=len(feasible),
            tpot_limit_ms=self._tpot_slo_ms,
            tpot_satisfied=(
                estimate.power_limit_feasible and estimate.decode_latency_s * 1e3 <= self._tpot_slo_ms
            ),
            reason=reason,
            decision_time_ms=(time.perf_counter() - start) * 1e3,
            prefill_tokens=state.prefill_tokens,
            decode_batch=state.decode_batch,
            decode_context_tokens=state.decode_context_tokens,
            remaining_layers=state.remaining_layers,
            prefill_layers=prefill_layers,
            decode_layers=total_layers,
        )
