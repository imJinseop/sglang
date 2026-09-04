from dataclasses import dataclass, field
import json
import os
from typing import List

import torch
import yaml

STREAM_GROUPS = []
SM_COUNTS = []
SM_GROUP_NUM = 8  # Default number of SM groups
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None


@dataclass
class PDMuxConfig:
    sm_group_num: int = 8
    manual_divisions: List[List[int]] = field(
        default_factory=list
    )  # [prefill_sm, decode_sm, decode_bs_threshold]
    split_forward_token_budget: int = 65536
    decode_bs_divisor: int = 36
    decode_graph_mode: str = "per_partition"


PDMUX_CONCURRENCY_MODES = {
    "per_partition",
    "fixed_partition",
    "single_decode_graph",
}


def apply_concurrency_mode(raw: dict) -> dict:
    """Apply the benchmark's process-wide concurrency ablation override."""
    result = dict(raw)
    manual_override = os.environ.get("SGLANG_PDMUX_MANUAL_DIVISIONS")
    if manual_override:
        manual_divisions = json.loads(manual_override)
        if not isinstance(manual_divisions, list) or not manual_divisions:
            raise ValueError("SGLANG_PDMUX_MANUAL_DIVISIONS must be a non-empty JSON list")
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


def load_pdmux_config(config_path: str) -> PDMuxConfig:
    """Load pdmux configuration from YAML file into a dataclass."""
    if not config_path:
        return PDMuxConfig()

    with open(config_path, "r") as f:
        raw = apply_concurrency_mode(yaml.safe_load(f) or {})

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")

    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be >= 3")

    manual_divisions = raw.get("manual_divisions", [])
    decode_graph_mode = raw.get("decode_graph_mode", "per_partition")
    if decode_graph_mode not in {"per_partition", "full_sm_once"}:
        raise ValueError(
            "decode_graph_mode must be one of: per_partition, full_sm_once"
        )

    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )

    return PDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        split_forward_token_budget=raw.get("split_forward_token_budget", 65536),
        decode_bs_divisor=raw.get("decode_bs_divisor", 36),
        decode_graph_mode=decode_graph_mode,
    )


def get_arch_constraints(compute_capability):
    major, minor = compute_capability
    # green context constraints for different architectures
    if major == 6:
        return 1, 1  # min_per_part, multiple
    elif major == 7:
        return 2, 2
    elif major == 8:
        return 4, 2
    elif major == 9 and minor >= 0:
        return 8, 8
    else:
        raise ValueError(f"Unsupported compute capability: {major}.{minor}")


def divide_sm(total_sms, compute_capability, groups):
    """
    :param total_sms: total sm count on a single GPU
    :param compute_capability: (major, minor)
    :return: SM partition group(prefill sm, decode sm)
    """
    min_per_part, multiple = get_arch_constraints(compute_capability)
    possible_values = [
        x
        for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
        if x >= total_sms - x and total_sms - x >= 16
    ]
    if not possible_values:
        raise ValueError(
            f"No valid partitions found for total SMs {total_sms} "
            f"with constraints (min per part: {min_per_part}, multiple: {multiple})"
        )

    if len(possible_values) >= groups:
        step = max(1, len(possible_values) // groups)
        selected_values = possible_values[::step][:groups]
    else:
        selected_values = possible_values

    divisions = []
    for part1 in selected_values:
        part2 = total_sms - part1
        divisions.append((part1, part2))

    divisions.reverse()  # Reverse to have larger prefill SM first

    return divisions


def initialize_stream_groups(gpu_id: int, config: PDMuxConfig):
    from sgl_kernel import spatial

    global STREAM_GROUPS, SM_COUNTS, SM_GROUP_NUM, CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    # for pd_multiplexing, Init stream_groups
    device = torch.cuda.current_device()
    total_sm_count = spatial.get_sm_available(gpu_id)
    # (prefill_sm_count, decode_sm_count)
    if config.manual_divisions:
        divisions = [
            (prefill_sm, decode_sm)
            for prefill_sm, decode_sm, _ in config.manual_divisions
        ]
    else:
        divisions = divide_sm(
            total_sm_count,
            torch.cuda.get_device_capability(device),
            config.sm_group_num - 2,
        )

    SM_COUNTS = []
    SM_COUNTS.append((total_sm_count, 0))  # Normal stream for prefill
    SM_COUNTS.extend(divisions)  # Add the divided SM counts
    SM_COUNTS.append((0, total_sm_count))  # Normal stream for decode
    STREAM_GROUPS = []
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for prefill
    for prefill_sm, decode_sm in divisions:
        STREAM_GROUPS.append(
            (spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id))
        )
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for decode
    if config.decode_graph_mode == "full_sm_once":
        full_decode_stream = STREAM_GROUPS[-1][1]
        STREAM_GROUPS = [
            (
                prefill_stream,
                (
                    full_decode_stream
                    if 0 < index < len(STREAM_GROUPS) - 1
                    else decode_stream
                ),
            )
            for index, (prefill_stream, decode_stream) in enumerate(STREAM_GROUPS)
        ]

    CURRENT_STREAM_IDX = 0
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def set_current_stream_idx(idx: int):
    global CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    if idx < 0 or idx >= len(STREAM_GROUPS):
        raise ValueError(f"Invalid stream index: {idx}")
    CURRENT_STREAM_IDX = idx
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def get_stream_groups() -> list[tuple[torch.cuda.Stream, torch.cuda.Stream]]:
    """Get the stream groups."""
    return STREAM_GROUPS


def get_sm_counts() -> list[tuple[int, int]]:
    """Get the SM counts."""
    return SM_COUNTS


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX


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
