"""Stable attachment boundary for the layerwise or an external PDMux policy.

The external policy can only select an already-created stream-group index.  It
cannot create a Green Context or introduce a new SM split at serving time.
"""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

from sglang.srt.multiplex.policy_state import PDMuxPolicyInput

if TYPE_CHECKING:
    from sglang.srt.multiplex.adaptive_pdmux_config import AdaptivePDMuxConfig


@dataclass(frozen=True)
class OursPolicyCandidate:
    stream_idx: int
    prefill_sm: int
    decode_sm: int


@dataclass(frozen=True)
class OursPolicyContext:
    objective: str
    tpot_slo_ms: float
    candidates: tuple[OursPolicyCandidate, ...]
    model_artifact_path: str | None = None
    frequency_mhz: int = 1410
    allow_extrapolation: bool = True
    hysteresis_fraction: float = 0.02


@dataclass(frozen=True)
class OursPolicyState:
    workload: PDMuxPolicyInput
    prefill_layers_in_quantum: int
    decode_layers_in_quantum: int

    @property
    def prefill_tokens(self) -> int:
        return self.workload.prefill.new_tokens

    @property
    def prefill_batch(self) -> int:
        return self.workload.prefill.batch_size

    @property
    def decode_batch(self) -> int:
        return self.workload.decode.batch_size

    @property
    def decode_context_tokens(self) -> int:
        return self.workload.decode.context_tokens

    @property
    def current_layer(self) -> int:
        return self.workload.prefill.current_layer

    @property
    def remaining_layers(self) -> int:
        return self.workload.prefill.remaining_layers


@dataclass(frozen=True)
class OursPolicySelection:
    stream_idx: int
    reason: str = "external_energy_policy"
    predicted_latency_ms: float | None = None
    predicted_energy_j: float | None = None
    predicted_power_w: float | None = None
    slo_satisfied: bool | None = None
    objective_score: float | None = None
    predicted_edp_j_s: float | None = None
    predicted_prefill_latency_ms: float | None = None
    predicted_decode_latency_ms: float | None = None
    predicted_contention_factor: float | None = None
    predicted_dram_bytes: float | None = None
    candidates_evaluated: int | None = None
    predicted_prefill_sm: int | None = None
    predicted_decode_sm: int | None = None


@dataclass(frozen=True)
class OursPDMuxDecision:
    selection: OursPolicySelection
    state: OursPolicyState
    decision_time_ms: float


class OursPolicy(Protocol):
    def select(
        self, state: OursPolicyState
    ) -> OursPolicySelection | Mapping[str, object] | int: ...


def _load_factory(entrypoint: str):
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "ours policy entrypoint must use 'module:factory' syntax; "
            f"got {entrypoint!r}"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(
            f"ours policy factory {attribute!r} was not found in {module_name!r}"
        ) from exc


class OursPDMuxPolicy:
    """Validate and normalize decisions from the configured cost model."""

    def __init__(
        self,
        config: AdaptivePDMuxConfig,
        sm_counts: list[tuple[int, int]],
    ) -> None:
        total_sm = sum(sm_counts[1])
        candidates = tuple(
            OursPolicyCandidate(
                index,
                prefill_sm,
                (
                    total_sm
                    if getattr(config, "decode_graph_mode", "per_partition")
                    == "full_sm_once"
                    else decode_sm
                ),
            )
            for index, (prefill_sm, decode_sm) in enumerate(sm_counts[1:-1], start=1)
        )
        if not candidates:
            raise ValueError("ours policy requires at least one paired partition")
        self.context = OursPolicyContext(
            objective=config.ours_objective,
            tpot_slo_ms=config.ours_tpot_slo_ms,
            candidates=candidates,
            model_artifact_path=getattr(config, "ours_model_artifact_path", None),
            frequency_mhz=getattr(config, "ours_frequency_mhz", 1410),
            allow_extrapolation=getattr(config, "ours_allow_extrapolation", True),
            hysteresis_fraction=getattr(config, "ours_hysteresis_fraction", 0.02),
        )
        self._candidate_indices = {item.stream_idx for item in candidates}
        self._split_forward_token_budget = config.split_forward_token_budget

        entrypoint = config.ours_policy_entrypoint or os.environ.get(
            "SGLANG_PDMUX_OURS_POLICY"
        )
        if not entrypoint:
            raise ValueError(
                "partition_policy='ours_energy' needs ours_policy_entrypoint "
                "in its YAML or SGLANG_PDMUX_OURS_POLICY"
            )
        factory = _load_factory(entrypoint)
        self._policy = factory(self.context)
        if not callable(getattr(self._policy, "select", None)):
            raise TypeError(
                "ours policy factory must return an object with select(state)"
            )

    @property
    def candidate_count(self) -> int:
        return len(self.context.candidates)

    @staticmethod
    def _normalize_selection(raw) -> OursPolicySelection:
        if isinstance(raw, OursPolicySelection):
            return raw
        if isinstance(raw, int):
            return OursPolicySelection(stream_idx=raw)
        if isinstance(raw, Mapping):
            return OursPolicySelection(**raw)
        raise TypeError(
            "ours select(state) must return OursPolicySelection, a mapping, "
            "or a stream index"
        )

    def select(self, policy_input: PDMuxPolicyInput) -> OursPDMuxDecision:
        start = time.perf_counter()
        prefill = policy_input.prefill
        prefill_tokens = prefill.new_tokens
        remaining_layers = prefill.remaining_layers
        prefill_layers = min(
            remaining_layers,
            max(
                1,
                self._split_forward_token_budget // max(1, prefill_tokens),
            ),
        )
        state = OursPolicyState(
            workload=policy_input,
            prefill_layers_in_quantum=prefill_layers,
            decode_layers_in_quantum=prefill.total_layers,
        )
        selection = self._normalize_selection(self._policy.select(state))
        if selection.stream_idx not in self._candidate_indices:
            raise ValueError(
                "ours policy selected stream_idx="
                f"{selection.stream_idx}, but fixed paired candidates are "
                f"{sorted(self._candidate_indices)}"
            )
        return OursPDMuxDecision(
            selection=selection,
            state=state,
            decision_time_ms=(time.perf_counter() - start) * 1e3,
        )
