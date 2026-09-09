# MI project extension; not part of upstream SGLang.
"""Fit the published MuxWise estimator from offline JSONL observations.

Input rows use one of these forms::

  {"kind":"prefill_solo", "prefill_sm":92,
   "new_tokens":[2048], "reused_tokens":[0], "latency_ms":12.3}
  {"kind":"decode_solo", "decode_sm":16,
   "context_tokens":[2048,2048], "latency_ms":8.1}
  {"kind":"concurrent", "prefill_sm":92, "decode_sm":16,
   "new_tokens":[2048], "reused_tokens":[0],
   "context_tokens":[2048,2048], "decode_latency_ms":9.2}

The paper publishes the feature equations, but not the fitting algorithm.  This
reproduction uses non-negative least squares independently at each pre-created
SM count and records the maximum concurrent slowdown in every guard cell.  The
non-negative constraint encodes the physical monotonicity of each workload
term and prevents an out-of-domain request from producing a negative latency.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from utils.experiment_env import env_device_name

from .model import (
    ContentionGuard,
    GuardCell,
    MuxWiseProfile,
    PhaseCoefficients,
)

DEFAULT_TOKEN_BOUNDARIES = (2048, 8192, 32768, 131072)
DEFAULT_DECODE_BATCH_BOUNDARIES = (
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    160,
    192,
    224,
    256,
)


def _ints(raw: Mapping[str, object], key: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw[key])  # type: ignore[arg-type]
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{key} must be a non-empty list of non-negative ints")
    return values


def _prefill_features(raw: Mapping[str, object]) -> tuple[float, ...]:
    new = _ints(raw, "new_tokens")
    reused = _ints(raw, "reused_tokens")
    if len(new) != len(reused):
        raise ValueError("new_tokens and reused_tokens must have equal length")
    return (
        float(sum(value * value for value in new)),
        float(sum(n * r for n, r in zip(new, reused))),
        float(sum(new)),
        1.0,
    )


def _decode_features(raw: Mapping[str, object]) -> tuple[float, ...]:
    context = _ints(raw, "context_tokens")
    return (float(sum(context)), float(len(context)), 1.0)


def _fit_phase(
    phase: str,
    sm: int,
    rows: Sequence[Mapping[str, object]],
    *, relative_error: bool = False,
) -> PhaseCoefficients:
    feature_fn = _prefill_features if phase == "prefill" else _decode_features
    expected_rank = 4 if phase == "prefill" else 3
    if len(rows) < expected_rank:
        raise ValueError(
            f"{phase} sm={sm} needs at least {expected_rank} independent samples"
        )
    x = np.asarray([feature_fn(row) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float64)
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("solo latency_ms values must be finite and > 0")
    theta = _nonnegative_least_squares(x / y[:, None], np.ones_like(y)) if relative_error else _nonnegative_least_squares(x, y)
    _unconstrained, _residuals, rank, _singular = np.linalg.lstsq(x, y, rcond=None)
    if rank < expected_rank:
        raise ValueError(
            f"{phase} sm={sm} samples are rank deficient ({rank}/{expected_rank})"
        )
    predicted = x @ theta
    if np.any(predicted <= 0):
        raise ValueError(f"{phase} sm={sm} fit predicts non-positive training latency")
    mape = float(np.mean(np.abs(predicted - y) / y))
    return PhaseCoefficients(
        phase=phase,  # type: ignore[arg-type]
        sm=sm,
        theta=tuple(float(value) for value in theta),
        fit_samples=len(rows),
        fit_mape=mape,
    )


def _nonnegative_least_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the tiny NNLS problem without adding a SciPy runtime dependency.

    MuxWise has only three decode and four prefill features, so enumerating all
    active coefficient subsets is cheap and gives the exact least-squares
    optimum for every feasible non-negative face.  Columns are scaled before
    solving to avoid the large ``sum(n^2)`` feature dominating conditioning.
    """

    feature_count = x.shape[1]
    scales = np.linalg.norm(x, axis=0)
    scales[scales == 0] = 1.0
    normalized = x / scales
    best_theta: np.ndarray | None = None
    best_error = math.inf
    tolerance = 1e-12
    for mask in range(1, 1 << feature_count):
        active = [index for index in range(feature_count) if mask & (1 << index)]
        solution, _residuals, _rank, _singular = np.linalg.lstsq(
            normalized[:, active], y, rcond=None
        )
        if np.any(solution < -tolerance):
            continue
        candidate_scaled = np.zeros(feature_count, dtype=np.float64)
        candidate_scaled[active] = np.maximum(solution, 0.0)
        residual = normalized @ candidate_scaled - y
        error = float(residual @ residual)
        if error < best_error:
            best_error = error
            best_theta = candidate_scaled / scales
    if best_theta is None:
        raise ValueError("non-negative least-squares fit has no feasible solution")
    return best_theta


def fit_profile(
    rows: Iterable[Mapping[str, object]],
    *,
    model: str,
    device: str,
    total_sm: int,
    token_boundaries: tuple[int, ...] = DEFAULT_TOKEN_BOUNDARIES,
    decode_batch_boundaries: tuple[int, ...] = DEFAULT_DECODE_BATCH_BOUNDARIES,
    relative_error: bool = False,
) -> MuxWiseProfile:
    items = tuple(rows)
    solo_prefill: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    solo_decode: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    concurrent: list[Mapping[str, object]] = []
    for row in items:
        kind = row.get("kind")
        if kind == "prefill_solo":
            solo_prefill[int(row["prefill_sm"])].append(row)
        elif kind == "decode_solo":
            solo_decode[int(row["decode_sm"])].append(row)
        elif kind == "concurrent":
            concurrent.append(row)
        else:
            raise ValueError(f"unsupported observation kind: {kind!r}")
    if not solo_prefill or not solo_decode or not concurrent:
        raise ValueError("prefill_solo, decode_solo, and concurrent rows are required")

    prefill_models = tuple(
        _fit_phase("prefill", sm, phase_rows, relative_error=relative_error)
        for sm, phase_rows in sorted(solo_prefill.items())
    )
    decode_models = tuple(
        _fit_phase("decode", sm, phase_rows, relative_error=relative_error)
        for sm, phase_rows in sorted(solo_decode.items())
    )
    decode_by_sm = {item.sm: item for item in decode_models}
    empty_guard = ContentionGuard(
        prefill_new_boundaries=token_boundaries,
        prefill_reused_boundaries=token_boundaries,
        decode_batch_boundaries=decode_batch_boundaries,
        decode_reused_boundaries=token_boundaries,
        cells=(),
        default_max_slowdown=1.0,
    )
    grouped: dict[tuple[int, ...], list[float]] = defaultdict(list)
    all_slowdowns = []
    for row in concurrent:
        prefill_sm = int(row["prefill_sm"])
        decode_sm = int(row["decode_sm"])
        if prefill_sm + decode_sm != total_sm:
            raise ValueError("concurrent row partition does not sum to total_sm")
        decode_model = decode_by_sm.get(decode_sm)
        if decode_model is None:
            raise ValueError(f"no decode solo fit for concurrent decode_sm={decode_sm}")
        contexts = _ints(row, "context_tokens")
        solo_ms = decode_model.predict_decode_ms(contexts)
        measured_ms = float(row["decode_latency_ms"])
        if not math.isfinite(measured_ms) or measured_ms <= 0:
            raise ValueError("decode_latency_ms must be finite and > 0")
        slowdown = max(1.0, measured_ms / solo_ms)
        new = _ints(row, "new_tokens")
        reused = _ints(row, "reused_tokens")
        if len(new) != len(reused):
            raise ValueError("concurrent prefill vectors must be aligned")
        key = empty_guard.key(
            prefill_sm=prefill_sm,
            decode_sm=decode_sm,
            prefill_new_tokens=sum(new),
            prefill_reused_tokens=sum(reused),
            decode_batch=len(contexts),
            decode_reused_tokens=sum(contexts),
        )
        grouped[key].append(slowdown)
        all_slowdowns.append(slowdown)

    cells = tuple(
        GuardCell(
            prefill_sm=key[0],
            decode_sm=key[1],
            prefill_new_bucket=key[2],
            prefill_reused_bucket=key[3],
            decode_batch_bucket=key[4],
            decode_reused_bucket=key[5],
            max_slowdown=max(values),
            samples=len(values),
        )
        for key, values in sorted(grouped.items())
    )
    guard = ContentionGuard(
        prefill_new_boundaries=token_boundaries,
        prefill_reused_boundaries=token_boundaries,
        decode_batch_boundaries=decode_batch_boundaries,
        decode_reused_boundaries=token_boundaries,
        cells=cells,
        default_max_slowdown=max(all_slowdowns),
    )
    return MuxWiseProfile(
        model=model,
        device=device,
        total_sm=total_sm,
        prefill_models=prefill_models,
        decode_models=decode_models,
        contention_guard=guard,
        metadata={
            "provenance": "independent reproduction of MuxWise Eq. 1/2 and max guard",
            "observations": str(len(items)),
            "fit": ("relative_nonnegative_least_squares_per_phase_sm" if relative_error
                    else "nonnegative_least_squares_per_phase_sm"),
        },
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
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default=env_device_name())
    parser.add_argument("--total-sm", type=int)
    args = parser.parse_args()
    if args.total_sm is None:
        from device import Device

        args.total_sm = int(Device.load(args.device).specification.num_sm)
    profile = fit_profile(
        _load_jsonl(args.observations),
        model=args.model,
        device=args.device,
        total_sm=args.total_sm,
    )
    profile.save(args.output)
    print(json.dumps(profile.to_json(), indent=2))


if __name__ == "__main__":
    main()
