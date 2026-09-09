# MI project extension; not part of upstream SGLang.
"""Configuration owned by the opt-in adaptive PDMux runtime.

The upstream ``pdmux_context`` module intentionally remains unaware of these
fields.  Its four native fields are inherited so one YAML file can still
describe both the pre-created Green Context menu and an experimental policy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from sglang.srt.mi_ext.multiplex.context import (apply_concurrency_mode)
from sglang.srt.multiplex.pdmux_context import (PDMuxConfig)


@dataclass
class AdaptivePDMuxConfig(PDMuxConfig):
    runtime_mode: str = "native"
    partition_policy: str = "decode_bs"
    repartition_interval_chunks: int = 0
    partition_transition: str = "host_sync_always"
    mi_path: Optional[str] = None
    mi_device_name: str = ""
    mi_device_dir: Optional[str] = None
    mi_calibration_path: Optional[str] = None
    mi_dtype: str = "bf16"
    mi_frequency_mhz: Optional[int] = None
    mi_objective: str = "latency"
    mi_allow_modeled_throttling: bool = True
    mi_allow_extrapolation: bool = True
    bullet_srm_profile_path: Optional[str] = None
    bullet_device_name: str = ""
    bullet_device_dir: Optional[str] = None
    bullet_dtype: str = "bf16"
    bullet_reference_frequency_mhz: Optional[int] = None
    bullet_tpot_slo_ms: float = 10.0
    bullet_tpot_headroom: float = 1.0
    bullet_prefill_layers_per_step: int = 4
    bullet_online_calibration: bool = True
    bullet_online_calibration_weight: float = 0.25
    muxwise_profile_path: Optional[str] = None
    muxwise_tbt_slo_ms: float = 50.0
    muxwise_tbt_headroom: float = 1.0
    muxwise_online_guard_update: bool = True
    muxwise_allow_sm_interpolation: bool = False
    profile_fixed_stream_idx: int = 1
    profile_observation_path: Optional[str] = None
    profile_prefill_layers_per_step: int = 32


def read_runtime_mode(config_path: str) -> str:
    """Read only the dispatch bit needed before entering the native init."""
    if not config_path:
        return "native"
    with open(config_path, "r", encoding="utf-8") as file:
        raw = apply_concurrency_mode(yaml.safe_load(file) or {})
    runtime_mode = raw.get("runtime_mode", "native")
    if runtime_mode not in {"native", "improved"}:
        raise ValueError("runtime_mode must be one of: native, improved")
    return runtime_mode


def load_adaptive_pdmux_config(config_path: str) -> AdaptivePDMuxConfig:
    """Load and validate fields used only by the adaptive runtime."""
    if not config_path:
        return AdaptivePDMuxConfig()

    with open(config_path, "r", encoding="utf-8") as file:
        raw = apply_concurrency_mode(yaml.safe_load(file) or {})
    config_dir = Path(config_path).resolve().parent

    def resolve_config_path(value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (config_dir / path).resolve())

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")
    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be >= 3")

    manual_divisions = raw.get("manual_divisions", [])
    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )

    runtime_mode = raw.get("runtime_mode", "native")
    if runtime_mode not in {"native", "improved"}:
        raise ValueError("runtime_mode must be one of: native, improved")

    partition_policy = raw.get("partition_policy", "decode_bs")
    valid_policies = {
        "decode_bs",
        "muxwise_artifact",
        "muxwise_slo",
        "mi",
        "bullet_srm_tpot",
        "offline_profile",
    }
    if partition_policy not in valid_policies:
        raise ValueError(
            "partition_policy must be one of: " + ", ".join(sorted(valid_policies))
        )
    if runtime_mode == "native" and partition_policy not in {
        "decode_bs",
        "muxwise_artifact",
    }:
        raise ValueError(
            "runtime_mode='native' only supports the upstream decode-BS "
            "dispatcher; custom policies require runtime_mode='improved'"
        )

    repartition_interval_chunks = int(raw.get("repartition_interval_chunks", 0))
    if repartition_interval_chunks < 0:
        raise ValueError("repartition_interval_chunks must be >= 0")

    partition_transition = raw.get("partition_transition", "host_sync_always")
    if partition_transition not in {"host_sync_always", "host_sync", "event_handoff"}:
        raise ValueError(
            "partition_transition must be one of: host_sync_always, "
            "host_sync, event_handoff"
        )

    decode_graph_mode = raw.get("decode_graph_mode", "per_partition")
    if decode_graph_mode not in {"per_partition", "full_sm_once"}:
        raise ValueError(
            "decode_graph_mode must be one of: per_partition, full_sm_once"
        )

    mi_objective = raw.get("mi_objective", "latency")
    if mi_objective not in {"latency", "energy", "edp"}:
        raise ValueError("mi_objective must be one of: latency, energy, edp")
    mi_frequency_value = os.environ.get(
        "SGLANG_PDMUX_MI_FREQUENCY_MHZ", raw.get("mi_frequency_mhz")
    )
    mi_frequency_mhz = (
        None if mi_frequency_value is None else int(mi_frequency_value)
    )
    if mi_frequency_mhz is not None and mi_frequency_mhz <= 0:
        raise ValueError("mi_frequency_mhz must be > 0")

    bullet_frequency_value = os.environ.get(
        "SGLANG_PDMUX_BULLET_FREQUENCY_MHZ",
        raw.get("bullet_reference_frequency_mhz"),
    )
    bullet_reference_frequency_mhz = (
        None if bullet_frequency_value is None else int(bullet_frequency_value)
    )
    if partition_policy == "bullet_srm_tpot" and bullet_reference_frequency_mhz is None:
        raise ValueError(
            "bullet_srm_tpot requires a device-derived reference frequency"
        )
    if bullet_reference_frequency_mhz is not None and bullet_reference_frequency_mhz <= 0:
        raise ValueError("bullet_reference_frequency_mhz must be > 0")
    bullet_tpot_slo_ms = float(
        os.environ.get(
            "SGLANG_PDMUX_BULLET_TPOT_SLO_MS",
            raw.get("bullet_tpot_slo_ms", 10.0),
        )
    )
    if bullet_tpot_slo_ms <= 0:
        raise ValueError("bullet_tpot_slo_ms must be > 0")
    bullet_tpot_headroom = float(raw.get("bullet_tpot_headroom", 1.0))
    if not 0 < bullet_tpot_headroom <= 1:
        raise ValueError("bullet_tpot_headroom must be in (0, 1]")
    bullet_prefill_layers_per_step = int(os.environ.get(
        "SGLANG_PDMUX_BULLET_PREFILL_LAYERS", raw.get("bullet_prefill_layers_per_step", 4)))
    if bullet_prefill_layers_per_step <= 0:
        raise ValueError("bullet_prefill_layers_per_step must be > 0")
    bullet_online_calibration_weight = float(
        raw.get("bullet_online_calibration_weight", 0.25)
    )
    if not 0 < bullet_online_calibration_weight <= 1:
        raise ValueError("bullet_online_calibration_weight must be in (0, 1]")

    muxwise_tbt_slo_ms = float(
        os.environ.get(
            "SGLANG_PDMUX_MUXWISE_TBT_SLO_MS",
            raw.get("muxwise_tbt_slo_ms", 50.0),
        )
    )
    if muxwise_tbt_slo_ms <= 0:
        raise ValueError("muxwise_tbt_slo_ms must be > 0")
    muxwise_tbt_headroom = float(raw.get("muxwise_tbt_headroom", 1.0))
    if not 0 < muxwise_tbt_headroom <= 1:
        raise ValueError("muxwise_tbt_headroom must be in (0, 1]")

    profile_fixed_stream_idx = int(
        os.environ.get(
            "SGLANG_PDMUX_PROFILE_STREAM_IDX",
            raw.get("profile_fixed_stream_idx", 1),
        )
    )
    if profile_fixed_stream_idx <= 0:
        raise ValueError("profile_fixed_stream_idx must select a paired stream")
    profile_prefill_layers_per_step = int(
        os.environ.get(
            "SGLANG_PDMUX_PROFILE_PREFILL_LAYERS",
            raw.get("profile_prefill_layers_per_step", 32),
        )
    )
    if profile_prefill_layers_per_step <= 0:
        raise ValueError("profile_prefill_layers_per_step must be > 0")
    profile_observation_override = os.environ.get(
        "SGLANG_PDMUX_PROFILE_OBSERVATION_PATH"
    )
    profile_observation_path = (
        str(Path(profile_observation_override).expanduser().resolve())
        if profile_observation_override
        else resolve_config_path(raw.get("profile_observation_path"))
    )
    if partition_policy == "offline_profile" and not profile_observation_path:
        raise ValueError(
            "offline_profile requires profile_observation_path or "
            "SGLANG_PDMUX_PROFILE_OBSERVATION_PATH"
        )

    return AdaptivePDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        split_forward_token_budget=raw.get("split_forward_token_budget", 65536),
        decode_bs_divisor=raw.get("decode_bs_divisor", 36),
        runtime_mode=runtime_mode,
        partition_policy=partition_policy,
        repartition_interval_chunks=repartition_interval_chunks,
        partition_transition=partition_transition,
        decode_graph_mode=decode_graph_mode,
        mi_path=resolve_config_path(
            os.environ.get("SGLANG_PDMUX_MI_PATH", raw.get("mi_path"))
        ),
        mi_device_name=os.environ.get(
            "SGLANG_PDMUX_MI_DEVICE_NAME", raw.get("mi_device_name", "")
        ),
        mi_device_dir=resolve_config_path(raw.get("mi_device_dir")),
        mi_dtype=os.environ.get(
            "SGLANG_PDMUX_MI_DTYPE", raw.get("mi_dtype", "bf16")
        ),
        mi_frequency_mhz=mi_frequency_mhz,
        mi_objective=mi_objective,
        mi_allow_modeled_throttling=raw.get("mi_allow_modeled_throttling", True),
        mi_allow_extrapolation=raw.get("mi_allow_extrapolation", True),
        mi_calibration_path=resolve_config_path(raw.get("mi_calibration_path")),
        bullet_srm_profile_path=resolve_config_path(
            os.environ.get(
                "SGLANG_PDMUX_BULLET_PROFILE_PATH",
                raw.get("bullet_srm_profile_path"),
            )
        ),
        bullet_device_name=os.environ.get(
            "SGLANG_PDMUX_BULLET_DEVICE_NAME",
            raw.get("bullet_device_name", ""),
        ),
        bullet_device_dir=resolve_config_path(raw.get("bullet_device_dir")),
        bullet_dtype=os.environ.get(
            "SGLANG_PDMUX_BULLET_DTYPE", raw.get("bullet_dtype", "bf16")
        ),
        bullet_reference_frequency_mhz=bullet_reference_frequency_mhz,
        bullet_tpot_slo_ms=bullet_tpot_slo_ms,
        bullet_tpot_headroom=bullet_tpot_headroom,
        bullet_prefill_layers_per_step=bullet_prefill_layers_per_step,
        bullet_online_calibration=raw.get("bullet_online_calibration", True),
        bullet_online_calibration_weight=bullet_online_calibration_weight,
        muxwise_profile_path=resolve_config_path(
            os.environ.get(
                "SGLANG_PDMUX_MUXWISE_PROFILE_PATH",
                raw.get("muxwise_profile_path"),
            )
        ),
        muxwise_tbt_slo_ms=muxwise_tbt_slo_ms,
        muxwise_tbt_headroom=muxwise_tbt_headroom,
        muxwise_online_guard_update=raw.get("muxwise_online_guard_update", True),
        muxwise_allow_sm_interpolation=raw.get("muxwise_allow_sm_interpolation", False),
        profile_fixed_stream_idx=profile_fixed_stream_idx,
        profile_observation_path=profile_observation_path,
        profile_prefill_layers_per_step=profile_prefill_layers_per_step,
    )
