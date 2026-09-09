# MI project extension; not part of upstream SGLang.
"""Write measured PDMux phase quanta as a common JSONL dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class ProfileObservationWriter:
    """Line-buffered observation writer owned by one scheduler process."""

    def __init__(self, path: str, *, total_layers: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._sequence = 0
        self._total_layers = int(total_layers)

    @staticmethod
    def _prefill_vectors(batch: ScheduleBatch | None) -> tuple[list[int], list[int]]:
        if batch is None or batch.is_empty():
            return [], []
        extend_lens = getattr(batch, "extend_lens", None)
        prefix_lens = getattr(batch, "prefix_lens", None)
        if extend_lens and prefix_lens and len(extend_lens) == len(prefix_lens):
            return (
                [int(value) for value in extend_lens],
                [int(value) for value in prefix_lens],
            )
        new_tokens = []
        reused_tokens = []
        for request in batch.reqs:
            prefix = len(request.prefix_indices)
            extend_range = getattr(request, "extend_range", None)
            uncached = (
                int(extend_range.length)
                if extend_range is not None
                else max(0, len(request.origin_input_ids) - prefix)
            )
            new_tokens.append(uncached)
            reused_tokens.append(prefix)
        return new_tokens, reused_tokens

    @staticmethod
    def _decode_context(batch: ScheduleBatch | None) -> list[int]:
        if batch is None or batch.is_empty():
            return []
        if batch.seq_lens_cpu is not None:
            return [int(value) for value in batch.seq_lens_cpu.tolist()]
        return [
            len(request.origin_input_ids) + len(request.output_ids)
            for request in batch.reqs
        ]

    def snapshot(
        self,
        *,
        phase: str,
        stream_idx: int,
        prefill_sm: int,
        decode_sm: int,
        prefill_batch: ScheduleBatch | None,
        decode_batch: ScheduleBatch | None,
        prefill_layer_start: int | None = None,
        prefill_layers: int = 0,
    ) -> dict[str, object]:
        new_tokens, reused_tokens = self._prefill_vectors(prefill_batch)
        decode_context = self._decode_context(decode_batch)
        return {
            "schema_version": 1,
            "phase": phase,
            "stream_idx": int(stream_idx),
            "prefill_sm": int(prefill_sm),
            "decode_sm": int(decode_sm),
            "prefill_new_tokens": new_tokens,
            "prefill_reused_tokens": reused_tokens,
            "prefill_batch": len(new_tokens),
            "decode_context_tokens": decode_context,
            "decode_batch": len(decode_context),
            "prefill_layer_start": prefill_layer_start,
            "prefill_layers": int(prefill_layers),
            "total_layers": self._total_layers,
            "concurrent": bool(new_tokens and decode_context),
        }

    def record(self, snapshot: dict[str, object], latency_ms: float) -> None:
        payload = {
            "sequence": self._sequence,
            "monotonic_s": time.perf_counter(),
            **snapshot,
            "latency_ms": float(latency_ms),
        }
        self._sequence += 1
        self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
