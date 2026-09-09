# MI project extension; not part of upstream SGLang.
"""SLO-aware MuxWise dispatcher over SGLang's pre-created PDMux menu."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .model import MuxWiseProfile
from sglang.srt.mi_ext.multiplex.policy_state import PDMuxPolicyInput

if TYPE_CHECKING:
    from sglang.srt.mi_ext.multiplex.adaptive_pdmux_config import AdaptivePDMuxConfig


@dataclass(frozen=True)
class MuxWisePDMuxDecision:
    stream_idx: int
    prefill_sm: int
    decode_sm: int
    predicted_prefill_ms: float
    predicted_decode_solo_ms: float
    predicted_decode_worst_ms: float
    decode_guard: float
    tbt_limit_ms: float
    tbt_satisfied: bool
    reason: str
    candidates_evaluated: int
    feasible_candidates: int
    decision_time_ms: float
    prefill_tokens: int
    prefill_reused_tokens: int
    decode_batch: int
    decode_context_tokens: int
    remaining_layers: int
    prefill_layers: int
    decode_layers: int
    prefill_new_vector: tuple[int, ...]
    prefill_reused_vector: tuple[int, ...]
    decode_context_vector: tuple[int, ...]


class MuxWiseSLOPolicy:
    """Published MuxWise estimator and just-enough-decode-SM policy.

    The estimator requires offline coefficients for every fixed candidate.  It
    never creates or extrapolates to a new Green Context at runtime.
    """

    def __init__(
        self,
        config: AdaptivePDMuxConfig,
        sm_counts: list[tuple[int, int]],
    ) -> None:
        if not config.muxwise_profile_path:
            raise ValueError(
                "partition_policy='muxwise_slo' requires muxwise_profile_path; "
                "fit a Llama-3.1-8B/A100 profile with related_work.muxwise.fit_profile"
            )
        self._profile = MuxWiseProfile.load(config.muxwise_profile_path)
        total_sm = sum(sm_counts[1])
        if self._profile.total_sm != total_sm:
            raise ValueError(
                "MuxWise profile and PDMux GPU disagree on total SM count: "
                f"profile={self._profile.total_sm}, PDMux={total_sm}"
            )
        self._tbt_slo_ms = config.muxwise_tbt_slo_ms
        self._tbt_headroom = config.muxwise_tbt_headroom
        self._online_guard_update = config.muxwise_online_guard_update
        self._allow_sm_interpolation = getattr(
            config, "muxwise_allow_sm_interpolation", False
        )
        self._plans = tuple(sm_counts[1:-1])
        if not self._plans:
            raise ValueError("MuxWise policy requires at least one paired partition")
        self._stream_idx = {
            (prefill_sm, decode_sm): stream_idx
            for stream_idx, (prefill_sm, decode_sm) in enumerate(self._plans, start=1)
        }
        # Fail at startup, not under load, if the fitted profile misses a fixed
        # candidate that can be selected by the server.
        for prefill_sm, decode_sm in self._plans:
            self._profile._for_sm(
                self._profile.prefill_models,
                prefill_sm,
                allow_interpolation=self._allow_sm_interpolation,
            )
            self._profile._for_sm(
                self._profile.decode_models,
                decode_sm,
                allow_interpolation=self._allow_sm_interpolation,
            )

    @property
    def candidate_count(self) -> int:
        return len(self._plans)

    @property
    def effective_tbt_ms(self) -> float:
        return self._tbt_slo_ms * self._tbt_headroom

    def select(self, policy_input: PDMuxPolicyInput) -> MuxWisePDMuxDecision:
        start = time.perf_counter()
        prefill = policy_input.prefill
        decode = policy_input.decode
        prefill_new = prefill.new_tokens_by_request
        prefill_reused = prefill.reused_tokens_by_request
        decode_context = decode.context_tokens_by_request
        if not prefill_new or not decode_context:
            raise ValueError("MuxWise paired dispatch requires both prefill and decode")
        estimates = tuple(
            (
                prefill_sm,
                decode_sm,
                self._profile.estimate(
                    prefill_sm=prefill_sm,
                    decode_sm=decode_sm,
                    prefill_new_tokens=prefill_new,
                    prefill_reused_tokens=prefill_reused,
                    decode_context_tokens=decode_context,
                    allow_sm_interpolation=self._allow_sm_interpolation,
                ),
            )
            for prefill_sm, decode_sm in self._plans
        )
        feasible = tuple(
            item
            for item in estimates
            if item[2].decode_worst_case_ms <= self.effective_tbt_ms
        )
        if feasible:
            selected = min(
                feasible,
                key=lambda item: (
                    item[1],
                    item[2].prefill_solo_ms,
                    item[2].decode_worst_case_ms,
                ),
            )
            satisfied = True
            reason = "minimum_decode_sm_satisfying_guarded_tbt"
        else:
            selected = min(
                estimates,
                key=lambda item: (item[2].decode_worst_case_ms, -item[1]),
            )
            satisfied = False
            reason = "tbt_overload_best_guarded_decode_candidate"
        prefill_sm, decode_sm, estimate = selected
        total_layers = prefill.total_layers
        remaining_layers = prefill.remaining_layers
        # MuxWise Eq. in SS3.4.2.  The phase regressions predict complete-model
        # latency, so this aligns a layer quantum with one guarded decode step.
        prefill_layers = min(
            remaining_layers,
            max(
                1,
                math.ceil(
                    estimate.decode_worst_case_ms
                    * total_layers
                    / estimate.prefill_solo_ms
                ),
            ),
        )
        return MuxWisePDMuxDecision(
            stream_idx=self._stream_idx[(prefill_sm, decode_sm)],
            prefill_sm=prefill_sm,
            decode_sm=decode_sm,
            predicted_prefill_ms=estimate.prefill_solo_ms,
            predicted_decode_solo_ms=estimate.decode_solo_ms,
            predicted_decode_worst_ms=estimate.decode_worst_case_ms,
            decode_guard=estimate.decode_guard,
            tbt_limit_ms=self.effective_tbt_ms,
            tbt_satisfied=satisfied,
            reason=reason,
            candidates_evaluated=len(estimates),
            feasible_candidates=len(feasible),
            decision_time_ms=(time.perf_counter() - start) * 1e3,
            prefill_tokens=sum(prefill_new),
            prefill_reused_tokens=sum(prefill_reused),
            decode_batch=len(decode_context),
            decode_context_tokens=sum(decode_context),
            remaining_layers=remaining_layers,
            prefill_layers=prefill_layers,
            decode_layers=total_layers,
            prefill_new_vector=prefill_new,
            prefill_reused_vector=prefill_reused,
            decode_context_vector=decode_context,
        )

    def observe_decode(
        self,
        decision: MuxWisePDMuxDecision,
        observed_decode_ms: float,
    ) -> bool:
        """Apply MuxWise's monotonic online contention-guard refinement.

        Returns whether the in-memory profile was updated.  Persistence is
        intentionally left to the experiment harness so measured runs never
        silently overwrite their offline input profile.
        """

        if not self._online_guard_update:
            return False
        observed_slowdown = observed_decode_ms / decision.predicted_decode_solo_ms
        refined = self._profile.contention_guard.refined(
            prefill_sm=decision.prefill_sm,
            decode_sm=decision.decode_sm,
            prefill_new_tokens=decision.prefill_tokens,
            prefill_reused_tokens=decision.prefill_reused_tokens,
            decode_batch=decision.decode_batch,
            decode_reused_tokens=decision.decode_context_tokens,
            observed_slowdown=observed_slowdown,
        )
        changed = refined != self._profile.contention_guard
        if changed:
            self._profile = replace(self._profile, contention_guard=refined)
        return changed
