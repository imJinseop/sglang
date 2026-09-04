"""Fit MuxWise and Bullet profiles from common fixed-partition observations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from utils.experiment_env import device_max_graphics_clock_mhz, env_device_name
from typing import Mapping

from ..bullet.fit_profile import fit_profile as fit_bullet_profile
from ..bullet.srm import SRMHardwareProfile
from ..bullet.types import ExecutionPlan, ExecutionState
from ..bullet.workloads import build_phase_layer_work
from ..muxwise.fit_profile import fit_profile as fit_muxwise_profile


def load_observations(paths: list[Path]) -> tuple[Mapping[str, object], ...]:
    rows = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(raw)
    if not rows:
        raise ValueError("no profile observations were loaded")
    return tuple(rows)


def muxwise_rows(
    rows: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    converted = []
    for row in rows:
        phase = row["phase"]
        concurrent = bool(row["concurrent"])
        if phase == "prefill" and not concurrent:
            if int(row["prefill_layers"]) != int(row["total_layers"]):
                continue
            converted.append(
                {
                    "kind": "prefill_solo",
                    "prefill_sm": int(row["prefill_sm"]),
                    "new_tokens": row["prefill_new_tokens"],
                    "reused_tokens": row["prefill_reused_tokens"],
                    "latency_ms": float(row["latency_ms"]),
                }
            )
        elif phase == "decode" and not concurrent:
            converted.append(
                {
                    "kind": "decode_solo",
                    "decode_sm": int(row["decode_sm"]),
                    "context_tokens": row["decode_context_tokens"],
                    "latency_ms": float(row["latency_ms"]),
                }
            )
        elif phase == "decode" and concurrent:
            converted.append(
                {
                    "kind": "concurrent",
                    "prefill_sm": int(row["prefill_sm"]),
                    "decode_sm": int(row["decode_sm"]),
                    "new_tokens": row["prefill_new_tokens"],
                    "reused_tokens": row["prefill_reused_tokens"],
                    "context_tokens": row["decode_context_tokens"],
                    "decode_latency_ms": float(row["latency_ms"]),
                }
            )
    return tuple(converted)


def bullet_rows(
    rows: tuple[Mapping[str, object], ...],
    *,
    model: str,
    dtype: str,
    frequency_mhz: int,
) -> tuple[Mapping[str, object], ...]:
    converted = []
    for row in rows:
        phase = str(row["phase"])
        prefill_new = tuple(int(value) for value in row["prefill_new_tokens"])
        decode_context = tuple(int(value) for value in row["decode_context_tokens"])
        total_layers = int(row["total_layers"])
        state = ExecutionState(
            prefill_tokens=sum(prefill_new),
            prefill_batch=len(prefill_new),
            decode_batch=len(decode_context),
            decode_context_tokens=sum(decode_context),
            remaining_layers=max(
                1,
                total_layers - int(row.get("prefill_layer_start") or 0),
            ),
        )
        plan = ExecutionPlan(
            prefill_sm=int(row["prefill_sm"]),
            decode_sm=int(row["decode_sm"]),
            frequency_mhz=frequency_mhz,
        )
        layer_work = build_phase_layer_work(phase, state, model=model, dtype=dtype)
        layers = total_layers if phase == "decode" else int(row["prefill_layers"])
        work = layer_work.scaled(layers)
        converted.append(
            {
                "phase": phase,
                "state": {
                    "prefill_tokens": state.prefill_tokens,
                    "prefill_batch": state.prefill_batch,
                    "decode_batch": state.decode_batch,
                    "decode_context_tokens": state.decode_context_tokens,
                    "remaining_layers": state.remaining_layers,
                },
                "plan": {
                    "prefill_sm": plan.prefill_sm,
                    "decode_sm": plan.decode_sm,
                    "frequency_mhz": plan.frequency_mhz,
                },
                "kernels": [
                    {
                        "name": kernel.name,
                        "flops": kernel.flops,
                        "dram_bytes": kernel.dram_bytes,
                    }
                    for kernel in work.kernels
                ],
                "latency_ms": float(row["latency_ms"]),
                "source": ("online_concurrent" if row["concurrent"] else "online_solo"),
            }
        )
    return tuple(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--target",
        choices=("both", "muxwise", "bullet"),
        default="both",
        help="Fit only the selected related-work model.",
    )
    parser.add_argument("--muxwise-output", type=Path)
    parser.add_argument("--bullet-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--mi-path", default=str(Path.cwd()))
    parser.add_argument(
        "--model-path",
        required=True,
        help="model directory, config.json path, or locally cached HF model ID",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default=env_device_name())
    parser.add_argument(
        "--frequency-mhz",
        type=int,
        default=device_max_graphics_clock_mhz(),
        help="defaults to the GPU selected by .env DEVICE_INDEX",
    )
    parser.add_argument("--total-sm", type=int)
    args = parser.parse_args()
    if args.total_sm is None:
        from device import Device

        args.total_sm = int(Device.load(args.device).specification.num_sm)
    fit_muxwise = args.target in {"both", "muxwise"}
    fit_bullet = args.target in {"both", "bullet"}
    if fit_muxwise and args.muxwise_output is None:
        parser.error("--muxwise-output is required for --target muxwise/both")
    if fit_bullet and args.bullet_output is None:
        parser.error("--bullet-output is required for --target bullet/both")
    if fit_bullet and args.mi_path not in sys.path:
        sys.path.insert(0, args.mi_path)

    observations = load_observations(args.observations)
    summary = {
        "target": args.target,
        "raw_observations": len(observations),
        "raw_by_phase_and_concurrency": {
            f"{phase}:{concurrent}": count
            for (phase, concurrent), count in Counter(
                (str(row["phase"]), bool(row["concurrent"])) for row in observations
            ).items()
        },
    }
    if fit_muxwise:
        mux_rows = muxwise_rows(observations)
        mux_profile = fit_muxwise_profile(
            mux_rows,
            model=args.model_path,
            device=args.device,
            total_sm=args.total_sm,
        )
        args.muxwise_output.parent.mkdir(parents=True, exist_ok=True)
        mux_profile.save(args.muxwise_output)
        summary.update(
            {
                "muxwise_rows": Counter(str(row["kind"]) for row in mux_rows),
                "muxwise_prefill_fit_mape": {
                    str(model.sm): model.fit_mape
                    for model in mux_profile.prefill_models
                },
                "muxwise_decode_fit_mape": {
                    str(model.sm): model.fit_mape for model in mux_profile.decode_models
                },
                "muxwise_guard_cells": len(mux_profile.contention_guard.cells),
                "muxwise_output": str(args.muxwise_output),
            }
        )
    if fit_bullet:
        bullet_input = bullet_rows(
            observations,
            model=args.model_path,
            dtype=args.dtype,
            frequency_mhz=args.frequency_mhz,
        )
        hardware = SRMHardwareProfile.from_device_name(
            args.device,
            dtype=args.dtype,
            reference_frequency_mhz=args.frequency_mhz,
        )
        bullet_profile = fit_bullet_profile(hardware, bullet_input)
        args.bullet_output.parent.mkdir(parents=True, exist_ok=True)
        bullet_profile.save(args.bullet_output)
        summary.update(
            {
                "bullet_profile_kind": "empirical_alpha_including_concurrency",
                "bullet_alpha_points": len(bullet_profile.calibration.points),
                "bullet_decode_contention_points": len(
                    bullet_profile.hardware.decode_contention
                ),
                "bullet_output": str(args.bullet_output),
            }
        )
    rendered = json.dumps(summary, indent=2, default=dict) + "\n"
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
