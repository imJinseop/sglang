"""Opt-in adaptive PDMux runtime extensions.

Nothing in this module executes for ``runtime_mode=native``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.distributed.parallel_state import set_pdmux_status
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.pdmux_context import (
    decode_execution_stream_idx,
    get_current_stream_idx,
    set_current_stream_idx,
)
from sglang.srt.multiplex.policy_state import build_policy_input

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerAdaptiveMultiplexMixin:

    def _init_pdmux_trace(self: Scheduler) -> None:
        self._pdmux_trace_file = None
        self._pdmux_trace_seq = 0
        self._pdmux_trace_start_s = time.perf_counter()
        trace_path = os.environ.get("SGLANG_PDMUX_TRACE_PATH")
        if not trace_path:
            return
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        self._pdmux_trace_file = open(trace_path, "w", encoding="utf-8", buffering=1)

    def _trace_pdmux_event(self: Scheduler, event: str, **fields) -> None:
        file = getattr(self, "_pdmux_trace_file", None)
        if file is None:
            return
        now = time.perf_counter()
        payload = {
            "seq": self._pdmux_trace_seq,
            "event": event,
            "monotonic_s": now,
            "elapsed_s": now - self._pdmux_trace_start_s,
            **fields,
        }
        self._pdmux_trace_seq += 1
        file.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _init_pdmux_observer(self: Scheduler) -> None:
        raw_interval = os.environ.get("SGLANG_PDMUX_OBSERVE_INTERVAL_S")
        self._pdmux_observe_interval_s = None
        self._pdmux_observe_last_s = float("-inf")
        self._pdmux_observe_start_s = time.perf_counter()
        self._pdmux_last_decision = {}
        if raw_interval is None:
            return
        try:
            interval_s = float(raw_interval)
        except ValueError as exc:
            raise ValueError(
                "SGLANG_PDMUX_OBSERVE_INTERVAL_S must be a positive number"
            ) from exc
        if interval_s <= 0:
            raise ValueError(
                "SGLANG_PDMUX_OBSERVE_INTERVAL_S must be a positive number"
            )
        self._pdmux_observe_interval_s = interval_s

    @staticmethod
    def _decode_context_tokens_for_observer(running_batch: ScheduleBatch) -> int:
        seq_lens_cpu = getattr(running_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            return int(seq_lens_cpu.sum().item())
        return sum(
            len(req.origin_input_ids) + len(req.output_ids)
            for req in running_batch.reqs
        )

    def _maybe_observe_pdmux_state(
        self: Scheduler,
        running_batch: ScheduleBatch,
        *,
        stream_idx: int,
        force: bool = False,
    ) -> None:
        """Print one compact, rate-limited development snapshot as JSON."""
        interval_s = getattr(self, "_pdmux_observe_interval_s", None)
        if interval_s is None:
            return
        now = time.perf_counter()
        if not force and now - self._pdmux_observe_last_s < interval_s:
            return

        decode_active = running_batch is not None and not running_batch.is_empty()
        prefill_batch = self.split_prefill_batch
        prefill_active = prefill_batch is not None and not prefill_batch.is_empty()
        if decode_active and prefill_active:
            phase = "concurrent"
        elif decode_active:
            phase = "decode_only"
        elif prefill_active:
            phase = "prefill_only"
        else:
            phase = "idle"

        self._pdmux_observe_last_s = now
        if phase == "idle" and not force:
            return

        decision = self._pdmux_last_decision
        payload = {
            "scheduler_elapsed_s": round(now - self._pdmux_observe_start_s, 3),
            "snapshot_reason": "partition_changed" if force else "periodic",
            "phase": phase,
            "stream_idx": stream_idx,
            "prefill_sm": self.sm_counts[stream_idx][0],
            "decode_sm": self.sm_counts[stream_idx][1],
            "decode_batch": running_batch.batch_size() if decode_active else 0,
            "decode_context_tokens": (
                self._decode_context_tokens_for_observer(running_batch)
                if decode_active
                else 0
            ),
            "prefill_batch": prefill_batch.batch_size() if prefill_active else 0,
            "prefill_tokens": (
                int(prefill_batch.extend_num_tokens) if prefill_active else 0
            ),
            "prefill_next_layer": (
                int(prefill_batch.split_index) if prefill_active else None
            ),
            "prefill_total_layers": self.model_config.num_hidden_layers,
            "policy": self.pdmux_config.partition_policy,
            "last_decision_trigger": decision.get("trigger"),
            "last_decision_changed": decision.get("changed"),
            "last_selected_stream_idx": decision.get("selected_stream_idx"),
            "last_predicted_decode_ms": decision.get("decode_latency_ms"),
            "last_predicted_prefill_ms": decision.get("prefill_latency_ms"),
            "last_predicted_blended_ms": decision.get("blended_latency_ms"),
            "last_predicted_tpot_ms": decision.get("predicted_tpot_ms"),
            "last_bottleneck_lane": decision.get("bottleneck_lane"),
        }
        print(
            "[PDMUX_STATE] " + json.dumps(payload, separators=(",", ":")),
            flush=True,
        )

    def init_adaptive_pdmux(self: Scheduler):
        """Initialize only the opt-in adaptive runtime extensions.

        Native PDMux owns config loading and Green Context construction.  This
        hook is called after those upstream steps and must never recreate a
        stream group or mutate the fixed candidate menu.
        """
        self.pdmux_chunk_count = 0
        self._init_pdmux_trace()
        self._init_pdmux_observer()
        self._pdmux_selected_prefill_layers = None
        self._pdmux_active_muxwise_decision = None
        self._pdmux_active_bullet_decision = None
        self._pdmux_pending_bullet_prefill_observations = []
        self._pdmux_pending_profile_prefill_observations = []
        self.pdmux_profile_writer = None
        self.mi_pdmux_policy = None
        self.bullet_pdmux_policy = None
        self.muxwise_slo_policy = None
        if self.pdmux_config.partition_policy == "mi":
            from sglang.srt.multiplex.mi_pdmux_policy import MIPDMuxPolicy

            self.mi_pdmux_policy = MIPDMuxPolicy(
                self.pdmux_config,
                self.sm_counts,
                model_config=self.model_config.hf_text_config,
                page_size=self.page_size,
            )
            logger.info(
                "MI PDMux policy initialized with %d pre-created partition "
                "candidates at %d MHz",
                self.mi_pdmux_policy.candidate_count,
                self.mi_pdmux_policy.frequency_mhz,
            )
        elif self.pdmux_config.partition_policy == "bullet_srm_tpot":
            from sglang.srt.multiplex.related_work.bullet.policy import (
                BulletSRMTPOTPolicy,
            )

            self.bullet_pdmux_policy = BulletSRMTPOTPolicy(
                self.pdmux_config,
                self.sm_counts,
                model_config=self.model_config.hf_text_config,
            )
            logger.info(
                "Bullet SRM TPOT policy initialized with %d fixed partition "
                "candidates at %d MHz, effective TPOT limit %.3f ms",
                self.bullet_pdmux_policy.candidate_count,
                self.bullet_pdmux_policy.frequency_mhz,
                self.bullet_pdmux_policy.effective_tpot_ms,
            )
        elif self.pdmux_config.partition_policy == "muxwise_slo":
            from sglang.srt.multiplex.related_work.muxwise.policy import (
                MuxWiseSLOPolicy,
            )

            self.muxwise_slo_policy = MuxWiseSLOPolicy(
                self.pdmux_config, self.sm_counts
            )
            logger.info(
                "MuxWise SLO policy initialized with %d fixed partition "
                "candidates, effective TBT limit %.3f ms",
                self.muxwise_slo_policy.candidate_count,
                self.muxwise_slo_policy.effective_tbt_ms,
            )
        elif self.pdmux_config.partition_policy == "offline_profile":
            from sglang.srt.multiplex.related_work.profiling import (
                ProfileObservationWriter,
            )

            stream_idx = self.pdmux_config.profile_fixed_stream_idx
            if not 0 < stream_idx < self.real_sm_group_num - 1:
                raise ValueError(
                    "profile_fixed_stream_idx must select one paired Green "
                    f"Context in [1, {self.real_sm_group_num - 2}]"
                )
            self.pdmux_profile_writer = ProfileObservationWriter(
                self.pdmux_config.profile_observation_path,
                total_layers=self.model_config.num_hidden_layers,
            )
            logger.info(
                "Offline PDMux profiler initialized: stream=%d sm=(P%d,D%d) "
                "layers_per_step=%d output=%s",
                stream_idx,
                self.sm_counts[stream_idx][0],
                self.sm_counts[stream_idx][1],
                self.pdmux_config.profile_prefill_layers_per_step,
                self.pdmux_config.profile_observation_path,
            )
        self._trace_pdmux_event(
            "pdmux_initialized",
            policy=self.pdmux_config.partition_policy,
            decode_graph_mode=self.pdmux_config.decode_graph_mode,
            repartition_interval_chunks=(self.pdmux_config.repartition_interval_chunks),
            split_forward_token_budget=(self.pdmux_config.split_forward_token_budget),
            partition_transition=self.pdmux_config.partition_transition,
            sm_counts=self.sm_counts,
        )

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups_adaptive(
        self: Scheduler,
        running_batch: ScheduleBatch,
        *,
        trigger: str,
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        """Select a stream group without mutating the active CUDA state.

        Keeping selection side-effect free lets the cost model run while work
        already queued on the current streams is still in flight.  The caller
        only drains those streams and activates a new group when the selected
        index actually differs from the current one.
        """
        previous_stream_idx = get_current_stream_idx()
        decision_fields = {}
        selected_prefill_layers = None
        if getattr(self, "pdmux_profile_writer", None) is not None and (
            not running_batch.is_empty() or self.split_prefill_batch is not None
        ):
            stream_idx = self.pdmux_config.profile_fixed_stream_idx
            selected_prefill_layers = (
                self.pdmux_config.profile_prefill_layers_per_step
                if self.split_prefill_batch is not None
                else None
            )
            decision_fields = {
                "profile_fixed_partition": True,
                "decode_batch": (
                    running_batch.batch_size() if not running_batch.is_empty() else 0
                ),
                "prefill_layers": selected_prefill_layers,
            }
        elif not running_batch.is_empty() and self.split_prefill_batch:
            policy_input = build_policy_input(
                running_batch=running_batch,
                prefill_batch=self.split_prefill_batch,
                total_layers=self.model_config.num_hidden_layers,
                current_stream_idx=previous_stream_idx,
                trigger=trigger,
            )
            if self.muxwise_slo_policy is not None:
                decision = self.muxwise_slo_policy.select(policy_input)
                stream_idx = decision.stream_idx
                selected_prefill_layers = decision.prefill_layers
                self._pdmux_active_muxwise_decision = decision
                decision_fields = {
                    "predicted_prefill_ms": decision.predicted_prefill_ms,
                    "predicted_decode_solo_ms": (decision.predicted_decode_solo_ms),
                    "predicted_decode_worst_ms": (decision.predicted_decode_worst_ms),
                    "decode_guard": decision.decode_guard,
                    "tbt_limit_ms": decision.tbt_limit_ms,
                    "tbt_satisfied": decision.tbt_satisfied,
                    "feasible_candidates": decision.feasible_candidates,
                    "candidates_evaluated": decision.candidates_evaluated,
                    "reason": decision.reason,
                    "decision_time_ms": decision.decision_time_ms,
                    "prefill_tokens": decision.prefill_tokens,
                    "prefill_reused_tokens": decision.prefill_reused_tokens,
                    "decode_batch": decision.decode_batch,
                    "decode_context_tokens": decision.decode_context_tokens,
                    "remaining_layers": decision.remaining_layers,
                    "prefill_layers": decision.prefill_layers,
                    "decode_layers": decision.decode_layers,
                }
                logger.info(
                    "MuxWise decision: chunk=%d layer=%d remaining=%d "
                    "quantum=(P_layers=%d,D_layers=%d) "
                    "state=(P_new=%d,P_reused=%d,D_bs=%d,D_context=%d) "
                    "stream=%d sm=(P%d,D%d) predicted=(P=%.3fms,"
                    "D_solo=%.3fms,D_worst=%.3fms,guard=%.3fx) "
                    "TBT_limit=%.3fms met=%s feasible=%d/%d reason=%s "
                    "decision=%.3fms",
                    self.pdmux_chunk_count,
                    self.split_prefill_batch.split_index,
                    decision.remaining_layers,
                    decision.prefill_layers,
                    decision.decode_layers,
                    decision.prefill_tokens,
                    decision.prefill_reused_tokens,
                    decision.decode_batch,
                    decision.decode_context_tokens,
                    decision.stream_idx,
                    decision.prefill_sm,
                    decision.decode_sm,
                    decision.predicted_prefill_ms,
                    decision.predicted_decode_solo_ms,
                    decision.predicted_decode_worst_ms,
                    decision.decode_guard,
                    decision.tbt_limit_ms,
                    decision.tbt_satisfied,
                    decision.feasible_candidates,
                    decision.candidates_evaluated,
                    decision.reason,
                    decision.decision_time_ms,
                )
            elif self.bullet_pdmux_policy is not None:
                decision = self.bullet_pdmux_policy.select(policy_input)
                stream_idx = decision.stream_idx
                selected_prefill_layers = decision.prefill_layers
                self._pdmux_active_bullet_decision = decision
                decision_fields = {
                    "predicted_tpot_ms": decision.predicted_tpot_ms,
                    "predicted_prefill_quantum_ms": (
                        decision.predicted_prefill_quantum_ms
                    ),
                    "predicted_remaining_prefill_ms": (
                        decision.predicted_remaining_prefill_ms
                    ),
                    "tpot_limit_ms": decision.tpot_limit_ms,
                    "tpot_satisfied": decision.tpot_satisfied,
                    "feasible_candidates": decision.feasible_candidates,
                    "candidates_evaluated": decision.candidates_evaluated,
                    "reason": decision.reason,
                    "decision_time_ms": decision.decision_time_ms,
                    "prefill_tokens": decision.prefill_tokens,
                    "decode_batch": decision.decode_batch,
                    "decode_context_tokens": decision.decode_context_tokens,
                    "remaining_layers": decision.remaining_layers,
                    "prefill_layers": decision.prefill_layers,
                    "decode_layers": decision.decode_layers,
                }
                logger.info(
                    "Bullet PDMux decision: chunk=%d layer=%d remaining=%d "
                    "quantum=(P_layers=%d,D_layers=%d) "
                    "state=(prefill_tokens=%d,decode_bs=%d,"
                    "decode_context_tokens=%d) stream=%d sm=(P%d,D%d) "
                    "f=%dMHz predicted=(TPOT=%.3fms,"
                    "P_quantum=%.3fms,P_remaining=%.3fms) "
                    "TPOT_limit=%.3fms met=%s feasible=%d/%d reason=%s "
                    "decision=%.3fms",
                    self.pdmux_chunk_count,
                    self.split_prefill_batch.split_index,
                    decision.remaining_layers,
                    decision.prefill_layers,
                    decision.decode_layers,
                    decision.prefill_tokens,
                    decision.decode_batch,
                    decision.decode_context_tokens,
                    decision.stream_idx,
                    decision.prefill_sm,
                    decision.decode_sm,
                    decision.frequency_mhz,
                    decision.predicted_tpot_ms,
                    decision.predicted_prefill_quantum_ms,
                    decision.predicted_remaining_prefill_ms,
                    decision.tpot_limit_ms,
                    decision.tpot_satisfied,
                    decision.feasible_candidates,
                    decision.candidates_evaluated,
                    decision.reason,
                    decision.decision_time_ms,
                )
            elif self.mi_pdmux_policy is not None:
                decision = self.mi_pdmux_policy.select(policy_input)
                stream_idx = decision.stream_idx
                selected_prefill_layers = decision.prefill_layers
                decision_fields = {
                    "decode_latency_ms": decision.decode_latency_ms,
                    "prefill_latency_ms": decision.prefill_latency_ms,
                    "blended_latency_ms": decision.blended_latency_ms,
                    "predicted_energy_j": decision.energy_j,
                    "predicted_power_w": decision.power_w,
                    "sustainable": decision.sustainable,
                    "throttle_factor": decision.throttle_factor,
                    "effective_frequency_mhz": decision.effective_frequency_mhz,
                    "selection_objective": self.pdmux_config.mi_objective,
                    "bottleneck_lane": (
                        "prefill"
                        if decision.prefill_latency_ms >= decision.decode_latency_ms
                        else "decode"
                    ),
                    "lane_latency_gap_ms": abs(
                        decision.prefill_latency_ms - decision.decode_latency_ms
                    ),
                    "candidates_evaluated": decision.candidates_evaluated,
                    "decision_time_ms": decision.decision_time_ms,
                    "prefill_tokens": decision.prefill_tokens,
                    "decode_batch": decision.decode_batch,
                    "decode_context_tokens": decision.decode_context_tokens,
                    "remaining_layers": decision.remaining_layers,
                    "prefill_layers": decision.prefill_layers,
                    "decode_layers": decision.decode_layers,
                }
                logger.info(
                    "MI PDMux decision: chunk=%d layer=%d remaining=%d "
                    "quantum=(P_layers=%d,D_layers=%d) "
                    "state=(prefill_tokens=%d, decode_bs=%d, "
                    "decode_context_tokens=%d) stream=%d sm=(P%d,D%d) "
                    "f=%dMHz predicted=(D=%.3fms,P=%.3fms,blend=%.3fms,"
                    "energy=%.3fJ,power=%.1fW,sustainable=%s,"
                    "throttle=%.3fx,effective_f=%.1fMHz) "
                    "candidates=%d decision=%.3fms",
                    self.pdmux_chunk_count,
                    self.split_prefill_batch.split_index,
                    decision.remaining_layers,
                    decision.prefill_layers,
                    decision.decode_layers,
                    decision.prefill_tokens,
                    decision.decode_batch,
                    decision.decode_context_tokens,
                    decision.stream_idx,
                    decision.prefill_sm,
                    decision.decode_sm,
                    decision.frequency_mhz,
                    decision.decode_latency_ms,
                    decision.prefill_latency_ms,
                    decision.blended_latency_ms,
                    decision.energy_j,
                    decision.power_w,
                    decision.sustainable,
                    decision.throttle_factor,
                    decision.effective_frequency_mhz,
                    decision.candidates_evaluated,
                    decision.decision_time_ms,
                )
            else:
                decode_bs = running_batch.batch_size()
                manual_divisions = self.pdmux_config.manual_divisions
                if manual_divisions:
                    stream_idx = 1
                    for i in range(len(manual_divisions)):
                        _, _, threshold = manual_divisions[i]
                        if decode_bs >= threshold:
                            stream_idx = i + 1
                else:
                    stream_idx = max(
                        1,
                        min(
                            self.real_sm_group_num - 2,
                            decode_bs
                            * (self.real_sm_group_num - 2)
                            // self.pdmux_config.decode_bs_divisor,
                        ),
                    )
                decision_fields = {
                    "decode_batch": decode_bs,
                    "decode_bs_divisor": self.pdmux_config.decode_bs_divisor,
                }
        elif not running_batch.is_empty():
            stream_idx = self.real_sm_group_num - 1
        else:
            stream_idx = 0

        self._pdmux_selected_prefill_layers = selected_prefill_layers
        if (
            getattr(self, "muxwise_slo_policy", None) is None
            or self.split_prefill_batch is None
        ):
            self._pdmux_active_muxwise_decision = None
        if (
            getattr(self, "bullet_pdmux_policy", None) is None
            or self.split_prefill_batch is None
        ):
            self._pdmux_active_bullet_decision = None

        changed = previous_stream_idx != stream_idx
        decode_backend_idx = decode_execution_stream_idx(
            self.pdmux_config, self.sm_counts, stream_idx
        )
        self._pdmux_last_decision = {
            "trigger": trigger,
            "previous_stream_idx": previous_stream_idx,
            "selected_stream_idx": stream_idx,
            "changed": changed,
            "effective_decode_sm": self.sm_counts[decode_backend_idx][1],
            **decision_fields,
        }
        self._trace_pdmux_event(
            "partition_decision",
            trigger=trigger,
            policy=self.pdmux_config.partition_policy,
            previous_stream_idx=previous_stream_idx,
            selected_stream_idx=stream_idx,
            changed=changed,
            prefill_sm=self.sm_counts[stream_idx][0],
            decode_sm=self.sm_counts[stream_idx][1],
            effective_decode_sm=self.sm_counts[decode_backend_idx][1],
            chunk=self.pdmux_chunk_count,
            layer=(
                self.split_prefill_batch.split_index
                if self.split_prefill_batch is not None
                else None
            ),
            **decision_fields,
        )
        return stream_idx, self.stream_groups[stream_idx]

    def _sync_streams_for_partition_change(
        self: Scheduler,
        prefill_stream: ExternalStream,
        decode_stream: ExternalStream,
        *,
        current_stream_idx: int,
        selected_stream_idx: int,
        trigger: str,
    ) -> bool:
        """Apply the configured ordering rule at a partition decision point.

        ``host_sync_always`` preserves public SGLang/MuxWise artifact behavior;
        ``host_sync`` skips the drain when the selected index is unchanged.
        ``event_handoff`` records dependencies on the old streams and makes the
        selected streams wait on-device, avoiding a CPU-blocking partition
        barrier.  This is not Bullet's libsmctrl oversubscription mechanism.
        """
        transition = self.pdmux_config.partition_transition
        partition_changed = selected_stream_idx != current_stream_idx
        if not partition_changed and transition != "host_sync_always":
            self._trace_pdmux_event(
                "partition_sync_skipped",
                trigger=trigger,
                stream_idx=current_stream_idx,
                reason="selected_partition_unchanged",
                prefill_sm=self.sm_counts[current_stream_idx][0],
                decode_sm=self.sm_counts[current_stream_idx][1],
            )
            return False

        if transition == "event_handoff":
            handoff_start = time.perf_counter()
            selected_prefill_stream, selected_decode_stream = self.stream_groups[
                selected_stream_idx
            ]
            prefill_event = getattr(self, "_pdmux_prefill_chunk_event", None)
            event_stream_idx = getattr(
                self, "_pdmux_prefill_chunk_event_stream_idx", None
            )
            reused_chunk_event = (
                trigger == "layer_chunk_boundary"
                and prefill_event is not None
                and event_stream_idx == current_stream_idx
            )
            if not reused_chunk_event:
                prefill_event = prefill_stream.record_event()
            decode_event = decode_stream.record_event()
            # Candidate Green Contexts can map the two lanes to different,
            # overlapping physical SM sets. Both new lanes must therefore
            # wait for both old lanes; lane-to-lane dependencies alone are not
            # a safe repartition barrier.
            selected_prefill_stream.wait_event(prefill_event)
            selected_prefill_stream.wait_event(decode_event)
            selected_decode_stream.wait_event(prefill_event)
            selected_decode_stream.wait_event(decode_event)
            self._trace_pdmux_event(
                "partition_event_handoff",
                trigger=trigger,
                from_stream_idx=current_stream_idx,
                to_stream_idx=selected_stream_idx,
                reused_prefill_chunk_event=reused_chunk_event,
                enqueue_ms=(time.perf_counter() - handoff_start) * 1e3,
            )
            return True

        sync_start = time.perf_counter()
        self._trace_pdmux_event(
            "partition_sync_start",
            trigger=trigger,
            from_stream_idx=current_stream_idx,
            to_stream_idx=selected_stream_idx,
            partition_changed=partition_changed,
            prefill_sm=self.sm_counts[current_stream_idx][0],
            decode_sm=self.sm_counts[current_stream_idx][1],
        )
        prefill_stream.synchronize()
        prefill_sync_end = time.perf_counter()
        decode_stream.synchronize()
        sync_end = time.perf_counter()
        self._trace_pdmux_event(
            "partition_sync_end",
            trigger=trigger,
            from_stream_idx=current_stream_idx,
            to_stream_idx=selected_stream_idx,
            partition_changed=partition_changed,
            prefill_sync_ms=(prefill_sync_end - sync_start) * 1e3,
            decode_sync_ms=(sync_end - prefill_sync_end) * 1e3,
            total_sync_ms=(sync_end - sync_start) * 1e3,
        )
        return partition_changed

    def _activate_stream_group(
        self: Scheduler, stream_idx: int
    ) -> tuple[ExternalStream, ExternalStream]:
        """Publish a selected group after its transition dependency is installed."""
        set_current_stream_idx(stream_idx)
        decode_backend_idx = decode_execution_stream_idx(
            self.pdmux_config, self.sm_counts, stream_idx
        )
        self.tp_worker.model_runner.update_decode_attn_backend(decode_backend_idx)
        return self.stream_groups[stream_idx]

    def update_split_prefill_batch(
        self: Scheduler, sm_count: int, running_batch: ScheduleBatch
    ) -> tuple[bool, ScheduleBatch]:
        if self.split_prefill_batch:
            return False, running_batch

        # add new request
        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch
        if batch and not batch.is_empty():
            batch.forward_mode = (
                ForwardMode.SPLIT_PREFILL
            )  # Set forward mode for split prefill
            self.split_prefill_batch = batch
            return True, running_batch
        return False, running_batch

    @torch.inference_mode()
    def event_loop_pdmux_adaptive(self: Scheduler):
        """Adaptive PDMux loop used only by ``runtime_mode=improved``."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        adjust_stream_group_reason = None
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        self._pdmux_prefill_chunk_event = None
        self._pdmux_prefill_chunk_event_stream_idx = None
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing...")

        while True:
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)
                running_batch = self.running_batch

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    created, running_batch = self.update_split_prefill_batch(
                        sm_count, running_batch=running_batch
                    )
                    self.running_batch = running_batch
                    if created:
                        adjust_stream_group = True
                        adjust_stream_group_reason = "new_split_prefill"
                        self._trace_pdmux_event(
                            "split_prefill_created",
                            prefill_tokens=self.split_prefill_batch.extend_num_tokens,
                            prefill_batch=self.split_prefill_batch.batch_size(),
                            decode_batch=running_batch.batch_size(),
                        )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                running_batch = self.update_running_batch(running_batch)
                self.running_batch = running_batch
                if stream_idx > 0 and running_batch.is_empty():
                    adjust_stream_group = True
                    adjust_stream_group_reason = "decode_batch_drained"
                if running_batch.is_empty() and self.split_prefill_batch is None:
                    self.on_idle()

            partition_changed = False
            if adjust_stream_group:
                trigger = adjust_stream_group_reason or "unspecified"
                selected_stream_idx, _ = self.adjust_stream_groups_adaptive(
                    running_batch=running_batch,
                    trigger=trigger,
                )
                changed = self._sync_streams_for_partition_change(
                    prefill_stream,
                    decode_stream,
                    current_stream_idx=stream_idx,
                    selected_stream_idx=selected_stream_idx,
                    trigger=trigger,
                )
                partition_changed = changed
                if changed:
                    previous_stream_idx = stream_idx
                    stream_idx = selected_stream_idx
                    stream_group = self._activate_stream_group(stream_idx)
                    prefill_stream = stream_group[0]
                    decode_stream = stream_group[1]
                    self._trace_pdmux_event(
                        "partition_applied",
                        trigger=trigger,
                        previous_stream_idx=previous_stream_idx,
                        selected_stream_idx=stream_idx,
                        prefill_sm=self.sm_counts[stream_idx][0],
                        decode_sm=self.sm_counts[stream_idx][1],
                    )
                    logger.info(
                        "Adjusted stream group: %d, prefill sm: %d, decode sm: %d",
                        stream_idx,
                        self.sm_counts[stream_idx][0],
                        self.sm_counts[stream_idx][1],
                    )
                adjust_stream_group = False
                adjust_stream_group_reason = None

            self._maybe_observe_pdmux_state(
                running_batch,
                stream_idx=stream_idx,
                force=partition_changed,
            )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if running_batch and not running_batch.is_empty():
                    timed_decision = (
                        self._pdmux_active_muxwise_decision
                        or self._pdmux_active_bullet_decision
                    )
                    profile_decode_snapshot = None
                    if self.pdmux_profile_writer is not None:
                        profile_decode_snapshot = self.pdmux_profile_writer.snapshot(
                            phase="decode",
                            stream_idx=stream_idx,
                            prefill_sm=self.sm_counts[stream_idx][0],
                            decode_sm=self.sm_counts[stream_idx][1],
                            prefill_batch=self.split_prefill_batch,
                            decode_batch=running_batch,
                            prefill_layer_start=(
                                self.split_prefill_batch.split_index
                                if self.split_prefill_batch is not None
                                else None
                            ),
                            prefill_layers=(self._pdmux_selected_prefill_layers or 0),
                        )
                    if (
                        timed_decision is not None
                        or profile_decode_snapshot is not None
                    ):
                        decode_timing_start = torch.cuda.Event(enable_timing=True)
                        decode_timing_end = torch.cuda.Event(enable_timing=True)
                        decode_timing_start.record(decode_stream)
                    else:
                        decode_timing_start = None
                        decode_timing_end = None
                    decode_result = self.run_batch(running_batch)
                    if decode_timing_end is not None:
                        decode_timing_end.record(decode_stream)
                    decode_done = True
                else:
                    profile_decode_snapshot = None
                    decode_timing_start = None
                    decode_timing_end = None
                    decode_done = False
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    policy_layers = self._pdmux_selected_prefill_layers
                    if policy_layers is not None:
                        forward_count = max(1, int(policy_layers))
                    else:
                        forward_count = (
                            max(
                                1,
                                self.pdmux_config.split_forward_token_budget
                                // self.split_prefill_batch.extend_num_tokens,
                            )
                            if self.split_prefill_batch.extend_num_tokens > 0
                            else self.model_config.num_hidden_layers
                        )
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    self._trace_pdmux_event(
                        "prefill_chunk_launch",
                        chunk=self.pdmux_chunk_count + 1,
                        layer_start=self.split_prefill_batch.split_index,
                        layer_end=next_split_index,
                        layer_count=forward_count,
                        prefill_tokens=self.split_prefill_batch.extend_num_tokens,
                        prefill_batch=self.split_prefill_batch.batch_size(),
                        decode_batch=running_batch.batch_size(),
                        stream_idx=stream_idx,
                        prefill_sm=self.sm_counts[stream_idx][0],
                        decode_sm=self.sm_counts[stream_idx][1],
                    )
                    bullet_prefill_decision = self._pdmux_active_bullet_decision
                    profile_prefill_snapshot = None
                    if self.pdmux_profile_writer is not None:
                        profile_prefill_snapshot = self.pdmux_profile_writer.snapshot(
                            phase="prefill",
                            stream_idx=stream_idx,
                            prefill_sm=self.sm_counts[stream_idx][0],
                            decode_sm=self.sm_counts[stream_idx][1],
                            prefill_batch=self.split_prefill_batch,
                            decode_batch=running_batch,
                            prefill_layer_start=(self.split_prefill_batch.split_index),
                            prefill_layers=forward_count,
                        )
                    if (
                        bullet_prefill_decision is not None
                        or profile_prefill_snapshot is not None
                    ):
                        prefill_timing_start = torch.cuda.Event(enable_timing=True)
                        prefill_timing_end = torch.cuda.Event(enable_timing=True)
                        prefill_timing_start.record(prefill_stream)
                    else:
                        prefill_timing_start = None
                        prefill_timing_end = None
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    if prefill_timing_end is not None:
                        prefill_timing_end.record(prefill_stream)
                        if bullet_prefill_decision is not None:
                            self._pdmux_pending_bullet_prefill_observations.append(
                                (
                                    prefill_timing_start,
                                    prefill_timing_end,
                                    bullet_prefill_decision,
                                )
                            )
                        if profile_prefill_snapshot is not None:
                            self._pdmux_pending_profile_prefill_observations.append(
                                (
                                    prefill_timing_start,
                                    prefill_timing_end,
                                    profile_prefill_snapshot,
                                )
                            )
                    interval = self.pdmux_config.repartition_interval_chunks
                    chunk_event = None
                    if (
                        interval > 0
                        or next_split_index == self.model_config.num_hidden_layers
                    ):
                        chunk_event = prefill_stream.record_event()
                        self._pdmux_prefill_chunk_event = chunk_event
                        self._pdmux_prefill_chunk_event_stream_idx = stream_idx
                        self._trace_pdmux_event(
                            "prefill_chunk_event_recorded",
                            chunk=self.pdmux_chunk_count + 1,
                            completed_through_layer=next_split_index,
                            stream_idx=stream_idx,
                            reason=(
                                "layer_boundary"
                                if next_split_index
                                < self.model_config.num_hidden_layers
                                else "prefill_finished"
                            ),
                        )
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = chunk_event
                    self.split_prefill_batch.split_index = next_split_index
                    self.pdmux_chunk_count += 1
                    self._trace_pdmux_event(
                        "prefill_chunk_enqueued",
                        chunk=self.pdmux_chunk_count,
                        completed_through_layer=next_split_index,
                        stream_idx=stream_idx,
                    )
                    if (
                        interval > 0
                        and next_split_index < self.model_config.num_hidden_layers
                        and self.pdmux_chunk_count % interval == 0
                    ):
                        adjust_stream_group = True
                        adjust_stream_group_reason = "layer_chunk_boundary"
                        self._trace_pdmux_event(
                            "partition_reevaluation_requested",
                            trigger=adjust_stream_group_reason,
                            chunk=self.pdmux_chunk_count,
                            next_layer=next_split_index,
                            current_stream_idx=stream_idx,
                        )
                        logger.info(
                            "PDMux layer-chunk boundary: completed chunk=%d, "
                            "next layer=%d; scheduling partition reevaluation",
                            self.pdmux_chunk_count,
                            next_split_index,
                        )

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_sync_start = time.perf_counter()
                decode_stream.synchronize()
                self._trace_pdmux_event(
                    "decode_step_sync",
                    stream_idx=stream_idx,
                    decode_ran=decode_done,
                    sync_ms=(time.perf_counter() - decode_sync_start) * 1e3,
                )
                if decode_done and decode_timing_start is not None:
                    observed_decode_ms = decode_timing_start.elapsed_time(
                        decode_timing_end
                    )
                    muxwise_decision = self._pdmux_active_muxwise_decision
                    if (
                        muxwise_decision is not None
                        and self.split_prefill_batch is not None
                    ):
                        updated = self.muxwise_slo_policy.observe_decode(
                            muxwise_decision, observed_decode_ms
                        )
                        self._trace_pdmux_event(
                            "muxwise_decode_observation",
                            observed_decode_ms=observed_decode_ms,
                            predicted_decode_solo_ms=(
                                muxwise_decision.predicted_decode_solo_ms
                            ),
                            predicted_decode_worst_ms=(
                                muxwise_decision.predicted_decode_worst_ms
                            ),
                            guard_updated=updated,
                            stream_idx=stream_idx,
                        )
                    bullet_decision = self._pdmux_active_bullet_decision
                    if bullet_decision is not None:
                        updated = self.bullet_pdmux_policy.observe_phase(
                            bullet_decision,
                            phase="decode",
                            observed_latency_ms=observed_decode_ms,
                        )
                        self._trace_pdmux_event(
                            "bullet_decode_observation",
                            observed_decode_ms=observed_decode_ms,
                            predicted_decode_ms=(bullet_decision.predicted_tpot_ms),
                            calibration_updated=updated,
                            stream_idx=stream_idx,
                        )
                    if profile_decode_snapshot is not None:
                        self.pdmux_profile_writer.record(
                            profile_decode_snapshot, observed_decode_ms
                        )
                if decode_done:
                    self.process_batch_result(running_batch, decode_result)

            pending_prefill = []
            for (
                start_event,
                end_event,
                bullet_decision,
            ) in self._pdmux_pending_bullet_prefill_observations:
                if not end_event.query():
                    pending_prefill.append((start_event, end_event, bullet_decision))
                    continue
                observed_prefill_ms = start_event.elapsed_time(end_event)
                updated = self.bullet_pdmux_policy.observe_phase(
                    bullet_decision,
                    phase="prefill",
                    observed_latency_ms=observed_prefill_ms,
                )
                self._trace_pdmux_event(
                    "bullet_prefill_observation",
                    observed_prefill_ms=observed_prefill_ms,
                    predicted_prefill_ms=(bullet_decision.predicted_prefill_quantum_ms),
                    calibration_updated=updated,
                    stream_idx=bullet_decision.stream_idx,
                )
            self._pdmux_pending_bullet_prefill_observations = pending_prefill

            pending_profile_prefill = []
            for (
                start_event,
                end_event,
                profile_snapshot,
            ) in self._pdmux_pending_profile_prefill_observations:
                if not end_event.query():
                    pending_profile_prefill.append(
                        (start_event, end_event, profile_snapshot)
                    )
                    continue
                self.pdmux_profile_writer.record(
                    profile_snapshot, start_event.elapsed_time(end_event)
                )
            self._pdmux_pending_profile_prefill_observations = pending_profile_prefill

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.ps.tp_size:
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        if running_batch and not running_batch.is_empty():
                            running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            running_batch = self.split_prefill_batch
                        self.running_batch = running_batch

                        self.split_prefill_batch = None
                        self.pdmux_chunk_count = 0
                        self._pdmux_prefill_chunk_event = None
                        self._pdmux_prefill_chunk_event_stream_idx = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
                        adjust_stream_group_reason = "split_prefill_finished"
                        self._trace_pdmux_event(
                            "split_prefill_finished",
                            next_decode_batch=running_batch.batch_size(),
                        )
