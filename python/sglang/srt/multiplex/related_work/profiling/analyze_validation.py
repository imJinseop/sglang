"""Summarize held-out PDMux cost-model validation traces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    items = tuple(float(value) for value in values)
    return {
        "samples": len(items),
        "mean": statistics.mean(items) if items else None,
        "p90": _percentile(items, 0.90),
        "p99": _percentile(items, 0.99),
        "max": max(items) if items else None,
    }


def _prediction_error(
    rows: Iterable[Mapping[str, object]],
    *,
    predicted_key: str,
    observed_key: str,
    slo_ms: float | None = None,
) -> dict[str, object]:
    items = tuple(rows)
    predicted = tuple(float(row[predicted_key]) for row in items)
    observed = tuple(float(row[observed_key]) for row in items)
    ape = tuple(abs(p - o) / o for p, o in zip(predicted, observed))
    result: dict[str, object] = {
        "samples": len(items),
        "mape": statistics.mean(ape) if ape else None,
        "ape_p90": _percentile(ape, 0.90),
        "ape_p99": _percentile(ape, 0.99),
        "mean_bias_ms": (
            statistics.mean(p - o for p, o in zip(predicted, observed))
            if items
            else None
        ),
        "underprediction_pct": (
            100 * sum(p < o for p, o in zip(predicted, observed)) / len(items)
            if items
            else None
        ),
    }
    if slo_ms is not None and items:
        result.update(
            {
                "predicted_slo_attainment_pct": (
                    100 * sum(value <= slo_ms for value in predicted) / len(items)
                ),
                "observed_gpu_quantum_slo_attainment_pct": (
                    100 * sum(value <= slo_ms for value in observed) / len(items)
                ),
            }
        )
    return result


def summarize(
    benchmark_path: Path,
    event_path: Path,
    *,
    slo_ms: float,
) -> dict[str, object]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    events = tuple(
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    paired_decisions = tuple(
        event
        for event in events
        if event.get("event") == "partition_decision"
        and int(event["prefill_sm"]) > 0
        and int(event["decode_sm"]) > 0
    )
    paired_applied = tuple(
        event
        for event in events
        if event.get("event") == "partition_applied"
        and int(event["prefill_sm"]) > 0
        and int(event["decode_sm"]) > 0
    )

    def plan_counts(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
        counts = Counter(
            f"P{int(row['prefill_sm'])}/D{int(row['decode_sm'])}" for row in rows
        )
        return dict(sorted(counts.items()))

    metrics = benchmark["metrics"]
    result: dict[str, object] = {
        "variant": benchmark["variant"],
        "scenario_sha256": benchmark["scenario"]["sha256"],
        "tpot_slo_ms": slo_ms,
        "end_to_end": {
            "completed_request_count": metrics["completed_request_count"],
            "load_phase_completed_requests": metrics["load_phase_completed_requests"],
            "normalized_request_throughput": metrics["normalized_request_throughput"],
            "normalized_output_throughput": metrics["normalized_output_throughput"],
            "drain_s": metrics["drain_s"],
            "ttft_ms": metrics["ttft_ms"],
            "tpot_ms": metrics["tpot_ms"],
            "tpot_slo_attainment_pct": metrics["tpot_slo_attainment_pct"],
            "energy_j": metrics["energy_j"],
            "joules_per_output_token": metrics["joules_per_output_token"],
        },
        "paired_partition_decisions": len(paired_decisions),
        "paired_decision_plan_counts": plan_counts(paired_decisions),
        "paired_partition_applications": len(paired_applied),
        "paired_applied_plan_counts": plan_counts(paired_applied),
        "decision_time_ms": _distribution(
            event["decision_time_ms"] for event in paired_decisions
        ),
        "event_handoff_enqueue_ms": _distribution(
            event["enqueue_ms"]
            for event in events
            if event.get("event") == "partition_event_handoff"
        ),
    }
    bullet_decode = tuple(
        event for event in events if event.get("event") == "bullet_decode_observation"
    )
    if bullet_decode:
        result["decode_gpu_quantum_prediction"] = _prediction_error(
            bullet_decode,
            predicted_key="predicted_decode_ms",
            observed_key="observed_decode_ms",
            slo_ms=slo_ms,
        )
        result["prefill_gpu_quantum_prediction"] = _prediction_error(
            (
                event
                for event in events
                if event.get("event") == "bullet_prefill_observation"
            ),
            predicted_key="predicted_prefill_ms",
            observed_key="observed_prefill_ms",
        )
    muxwise_decode = tuple(
        event for event in events if event.get("event") == "muxwise_decode_observation"
    )
    if muxwise_decode:
        result["decode_solo_prediction"] = _prediction_error(
            muxwise_decode,
            predicted_key="predicted_decode_solo_ms",
            observed_key="observed_decode_ms",
        )
        result["decode_guarded_prediction"] = _prediction_error(
            muxwise_decode,
            predicted_key="predicted_decode_worst_ms",
            observed_key="observed_decode_ms",
            slo_ms=slo_ms,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slo-ms", type=float, required=True)
    args = parser.parse_args()
    summary = summarize(args.benchmark, args.events, slo_ms=args.slo_ms)
    rendered = json.dumps(summary, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
