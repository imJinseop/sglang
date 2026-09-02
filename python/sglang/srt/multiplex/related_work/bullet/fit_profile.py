"""Build a Bullet SRM profile from sparse measured execution states.

Hardware JSON follows :class:`SRMHardwareProfile`. Observation JSONL rows use::

  {"phase":"decode",
   "state":{"prefill_tokens":2048,"prefill_batch":1,"decode_batch":8,
            "decode_context_tokens":8192,"remaining_layers":32},
   "plan":{"prefill_sm":84,"decode_sm":24,"frequency_mhz":1410},
   "kernels":[{"name":"qkv","flops":123,"dram_bytes":456}],
   "latency_ms":7.2}

Each row becomes Bullet Equation 2's
``alpha = measured_latency / SRM_roofline_latency``. The paper specifies sparse
sampling and linear extrapolation, but does not publish its precise sampler or
interpolation implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .srm import (
    BulletSRMPredictor,
    SRMCalibration,
    SRMCalibrationPoint,
    SRMHardwareProfile,
    execution_features,
)
from .types import ExecutionPlan, ExecutionState, KernelWork, LaneWork


def fit_profile(
    hardware: SRMHardwareProfile,
    rows: tuple[Mapping[str, object], ...],
) -> BulletSRMPredictor:
    if not rows:
        raise ValueError("at least one calibration row is required")
    bare = BulletSRMPredictor(hardware)
    points = []
    for row in rows:
        phase = str(row["phase"])
        if phase not in {"decode", "prefill"}:
            raise ValueError("phase must be 'decode' or 'prefill'")
        state = ExecutionState(**row["state"])  # type: ignore[arg-type]
        plan = ExecutionPlan(**row["plan"])  # type: ignore[arg-type]
        work = LaneWork(
            tuple(
                KernelWork(
                    name=str(kernel["name"]),
                    flops=float(kernel["flops"]),
                    dram_bytes=float(kernel["dram_bytes"]),
                )
                for kernel in row["kernels"]  # type: ignore[union-attr]
            )
        )
        latency_s = float(row["latency_ms"]) / 1e3
        roofline_s = bare.roofline_latency(phase, state, work, plan)  # type: ignore[arg-type]
        points.append(
            SRMCalibrationPoint(
                phase=phase,  # type: ignore[arg-type]
                features=execution_features(state, plan),
                scaling_factor=latency_s / roofline_s,
                source=str(row.get("source", "offline")),
            )
        )
    return BulletSRMPredictor(
        hardware,
        SRMCalibration(
            points=tuple(points),
            metadata={
                "provenance": "independent Bullet Eq. 1/2 reproduction",
                "observations": str(len(rows)),
            },
        ),
    )


def _load_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        rows.append(raw)
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hardware_raw = json.loads(args.hardware.read_text(encoding="utf-8"))
    predictor = fit_profile(
        SRMHardwareProfile.from_json(hardware_raw),
        _load_jsonl(args.observations),
    )
    predictor.save(args.output)
    print(json.dumps(predictor.to_json(), indent=2))


if __name__ == "__main__":
    main()
