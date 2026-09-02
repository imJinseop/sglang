"""Generate a deterministic mixed solo/concurrent profiling scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_group(
    requests: list[dict[str, object]],
    *,
    arrival_s: float,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
    prompt_group: int | None = None,
) -> None:
    for _ in range(batch_size):
        requests.append(
            {
                "request_id": len(requests),
                "arrival_s": arrival_s,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "prompt_group": prompt_group,
            }
        )


def build_scenario() -> dict[str, object]:
    requests: list[dict[str, object]] = []
    # Prefill-only rank coverage: max_new_tokens=1 terminates with the token
    # produced by prefill, so no decode iteration remains.
    for arrival, batch, tokens, prompt_group in (
        (0.0, 1, 128, 10),
        (0.8, 2, 256, 10),  # about 128 reused + 128 new
        (1.6, 1, 512, 20),
        (2.4, 2, 1024, 20),  # about 512 reused + 512 new
        (3.2, 1, 1024, 30),
        (4.2, 1, 3072, 30),  # about 1024 reused + 2048 new
        (5.5, 1, 4096, None),
        (7.0, 4, 768, None),
        (8.0, 2, 1536, None),
    ):
        add_group(
            requests,
            arrival_s=arrival,
            batch_size=batch,
            prompt_tokens=tokens,
            output_tokens=1,
            prompt_group=prompt_group,
        )

    # Unequal output lengths make the decode batch shrink through several
    # values, providing independent sum(context), batch, and intercept samples.
    for index in range(12):
        requests.append(
            {
                "request_id": len(requests),
                "arrival_s": 10.0,
                "prompt_tokens": (128, 256, 512, 1024)[index % 4],
                "output_tokens": 48 + 8 * index,
            }
        )

    # These prefill-only probes arrive while the long decode anchor is alive.
    # They produce concurrent latency/slowdown observations without joining the
    # subsequent decode batch themselves.
    for arrival, batch, tokens, prompt_group in (
        (10.25, 1, 256, 10),
        (10.55, 2, 1024, 20),
        (10.9, 1, 3072, 30),
        (11.3, 2, 512, None),
        (11.8, 1, 2048, None),
        (12.3, 2, 1024, None),
        (12.8, 1, 4096, None),
    ):
        add_group(
            requests,
            arrival_s=arrival,
            batch_size=batch,
            prompt_tokens=tokens,
            output_tokens=1,
            prompt_group=prompt_group,
        )

    requests.sort(key=lambda item: (item["arrival_s"], item["request_id"]))
    for request_id, request in enumerate(requests):
        request["request_id"] = request_id
    duration_s = 17.0
    return {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "target_qps": len(requests) / duration_s,
        "duration_s": duration_s,
        "source": "related_work_fixed_partition_profile_v1",
        "max_model_len": 8192,
        "token_generator": "mi_stride_v1",
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenario = build_scenario()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: {len(scenario['requests'])} requests, "
        f"duration={scenario['duration_s']}s"
    )


if __name__ == "__main__":
    main()
