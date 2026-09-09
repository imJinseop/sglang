# MI project extension; not part of upstream SGLang.
"""Concurrency configuration and stream ownership for MI PDMux experiments."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import torch
import yaml

if TYPE_CHECKING:
    from sglang.srt.multiplex.pdmux_context import PDMuxConfig

# Retain the original GC streams for the lifetime of their priority siblings.
OMNIDECODE_PREFILL_STREAM_OWNERS = []
PDMUX_CONCURRENCY_MODES = {"per_partition", "fixed_partition", "single_decode_graph"}


def _omnidecode_prefill_stream(stream, gpu_id):
    """Use the same GC resources with highest-priority eager prefill work."""
    from cuda.bindings import driver as cuda

    def checked(result):
        status, *values = result
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"OmniDecode stream creation failed: {status}")
        return values

    _, highest = checked(cuda.cuCtxGetStreamPriorityRange())
    (green_ctx,) = checked(cuda.cuStreamGetGreenCtx(stream.cuda_stream))
    (handle,) = checked(
        cuda.cuGreenCtxStreamCreate(
            green_ctx, int(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING), highest
        )
    )
    OMNIDECODE_PREFILL_STREAM_OWNERS.append(stream)
    return torch.cuda.get_stream_from_external(int(handle), device=gpu_id)


def apply_concurrency_mode(raw: dict) -> dict:
    """Apply the benchmark's process-wide concurrency ablation override."""
    result = dict(raw)
    manual_override = os.environ.get("SGLANG_PDMUX_MANUAL_DIVISIONS")
    if manual_override:
        manual_divisions = json.loads(manual_override)
        if not isinstance(manual_divisions, list) or not manual_divisions:
            raise ValueError(
                "SGLANG_PDMUX_MANUAL_DIVISIONS must be a non-empty JSON list"
            )
        result["manual_divisions"] = manual_divisions
        result["sm_group_num"] = len(manual_divisions) + 2
    mode = os.environ.get("SGLANG_PDMUX_CONCURRENCY_MODE", "per_partition")
    if mode not in PDMUX_CONCURRENCY_MODES:
        raise ValueError(
            "SGLANG_PDMUX_CONCURRENCY_MODE must be one of: "
            + ", ".join(sorted(PDMUX_CONCURRENCY_MODES))
        )
    if mode == "fixed_partition":
        raw_division = os.environ.get("SGLANG_PDMUX_FIXED_DIVISION", "")
        try:
            prefill_sm, decode_sm = (
                int(value.strip()) for value in raw_division.split(",", 1)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fixed_partition requires SGLANG_PDMUX_FIXED_DIVISION=PREFILL,DECODE"
            ) from exc
        if prefill_sm <= 0 or decode_sm <= 0:
            raise ValueError("fixed partition SM counts must be positive")
        result["sm_group_num"] = 3
        result["manual_divisions"] = [[prefill_sm, decode_sm, 1]]
        result["decode_graph_mode"] = "per_partition"
        result["muxwise_allow_sm_interpolation"] = True
    elif mode == "single_decode_graph":
        result["decode_graph_mode"] = "full_sm_once"
    return result


def read_decode_graph_mode(config_path: str) -> str:
    if not config_path:
        return "per_partition"
    with open(config_path, "r", encoding="utf-8") as file:
        raw = apply_concurrency_mode(yaml.safe_load(file) or {})
    mode = raw.get("decode_graph_mode", "per_partition")
    if mode not in {"per_partition", "full_sm_once"}:
        raise ValueError(
            "decode_graph_mode must be one of: per_partition, full_sm_once"
        )
    return mode


def decode_execution_stream_idx(
    config: PDMuxConfig,
    sm_counts: list[tuple[int, int]],
    selected_stream_idx: int,
) -> int:
    """Map a policy partition to the backend that executes decode.

    In ``full_sm_once`` mode decode is always planned and executed by the
    maximum-SM backend, including eager fallback batches that cannot use a
    captured graph. The selected index still chooses the eager prefill Green
    Context.
    """
    # Older/native configs and lightweight test doubles predate this option.
    # Their established behavior is one decode backend per selected partition.
    if getattr(config, "decode_graph_mode", "per_partition") != "full_sm_once":
        return selected_stream_idx
    return max(range(len(sm_counts)), key=lambda index: sm_counts[index][1])


def decode_graph_mode(raw: dict) -> str:
    mode = raw.get("decode_graph_mode", "per_partition")
    if mode not in {"per_partition", "full_sm_once"}:
        raise ValueError(
            "decode_graph_mode must be one of: per_partition, full_sm_once"
        )
    return mode


def configure_stream_groups(stream_groups, config, gpu_id):
    if config.decode_graph_mode != "full_sm_once":
        return stream_groups
    full_decode_stream = stream_groups[-1][1]
    return [
        (
            (
                _omnidecode_prefill_stream(prefill, gpu_id)
                if 0 < index < len(stream_groups) - 1
                else prefill
            ),
            full_decode_stream if 0 < index < len(stream_groups) - 1 else decode,
        )
        for index, (prefill, decode) in enumerate(stream_groups)
    ]
