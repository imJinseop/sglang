# MI project extension; not part of upstream SGLang.
"""Lift kernel work into one-layer Bullet scheduling quanta."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ExecutionState, LaneWork


@dataclass(frozen=True)
class BulletLaneWorkPair:
    decode: LaneWork
    prefill: LaneWork


def build_phase_layer_work(
    phase: str,
    state: ExecutionState,
    *,
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype: str = "bf16",
) -> LaneWork:
    """Build one phase's per-kernel work even when the other lane is idle."""

    from modeling.cost_model.operation_count_estimator.llm_composition import build_llm_workload

    if phase == "prefill":
        if state.prefill_tokens <= 0 or state.prefill_batch <= 0:
            raise ValueError("prefill work requires an active prefill batch")
        input_tokens = max(1, round(state.mean_prefill_sequence))
        batch_size = state.prefill_batch
        output_tokens = 0
    elif phase == "decode":
        if state.decode_batch <= 0:
            raise ValueError("decode work requires an active decode batch")
        input_tokens = max(1, round(state.mean_decode_context))
        batch_size = state.decode_batch
        output_tokens = 1
    else:
        raise ValueError("phase must be 'prefill' or 'decode'")
    workload = build_llm_workload(
        model=model,
        component="layer",
        stage=phase,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        dtype=dtype,
        batch_size=batch_size,
    )
    return LaneWork.from_workload(workload)


def build_llm_layer_work(
    state: ExecutionState,
    *,
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype: str = "bf16",
) -> BulletLaneWorkPair:
    """Return per-kernel work for one Transformer layer in each lane.

    MI's workload builder reads the serving model config and emits one layer,
    which Bullet's controller multiplies by its next prefill quantum or a full
    decode iteration. MI supplies kernel work only; the SRM and scheduling rule
    remain independent related-work implementations in this package.
    """

    if state.prefill_tokens <= 0 or state.decode_batch <= 0:
        raise ValueError("concurrent work requires active prefill and decode")
    return BulletLaneWorkPair(
        decode=build_phase_layer_work("decode", state, model=model, dtype=dtype),
        prefill=build_phase_layer_work("prefill", state, model=model, dtype=dtype),
    )
