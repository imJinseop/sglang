"""CUDA-free input contract shared by adaptive PDMux policies.

The scheduler translates SGLang's mutable ``ScheduleBatch`` objects once at a
decision boundary.  Cost models consume these immutable value objects instead
of depending on scheduler internals or receiving a long list of scalar
arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass(frozen=True)
class PrefillPolicyState:
    new_tokens_by_request: tuple[int, ...]
    reused_tokens_by_request: tuple[int, ...]
    current_layer: int
    total_layers: int

    @property
    def batch_size(self) -> int:
        return len(self.new_tokens_by_request)

    @property
    def new_tokens(self) -> int:
        return sum(self.new_tokens_by_request)

    @property
    def reused_tokens(self) -> int:
        return sum(self.reused_tokens_by_request)

    @property
    def remaining_layers(self) -> int:
        return max(1, self.total_layers - self.current_layer)


@dataclass(frozen=True)
class DecodePolicyState:
    context_tokens_by_request: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.context_tokens_by_request)

    @property
    def context_tokens(self) -> int:
        return sum(self.context_tokens_by_request)


@dataclass(frozen=True)
class PDMuxPolicyInput:
    prefill: PrefillPolicyState
    decode: DecodePolicyState
    current_stream_idx: int
    trigger: str


def _prefill_vectors(batch: ScheduleBatch) -> tuple[tuple[int, ...], tuple[int, ...]]:
    extend_lens = getattr(batch, "extend_lens", None)
    prefix_lens = getattr(batch, "prefix_lens", None)
    if extend_lens and prefix_lens and len(extend_lens) == len(prefix_lens):
        return (
            tuple(int(value) for value in extend_lens),
            tuple(int(value) for value in prefix_lens),
        )

    new_tokens = []
    reused_tokens = []
    for request in batch.reqs:
        reused = len(request.prefix_indices)
        extend_range = getattr(request, "extend_range", None)
        if extend_range is not None:
            new = int(extend_range.length)
        else:
            new = max(0, len(request.origin_input_ids) - reused)
        new_tokens.append(new)
        reused_tokens.append(reused)

    if not new_tokens:
        # Defensive fallback for synthetic/test batches without request objects.
        total = int(getattr(batch, "extend_num_tokens", 0))
        new_tokens.append(total)
        reused_tokens.append(0)
    return tuple(new_tokens), tuple(reused_tokens)


def _decode_context(batch: ScheduleBatch) -> tuple[int, ...]:
    if batch.seq_lens_cpu is not None:
        return tuple(int(value) for value in batch.seq_lens_cpu.tolist())
    return tuple(
        len(request.origin_input_ids) + len(request.output_ids)
        for request in batch.reqs
    )


def build_policy_input(
    *,
    running_batch: ScheduleBatch,
    prefill_batch: ScheduleBatch,
    total_layers: int,
    current_stream_idx: int,
    trigger: str,
) -> PDMuxPolicyInput:
    new_tokens, reused_tokens = _prefill_vectors(prefill_batch)
    return PDMuxPolicyInput(
        prefill=PrefillPolicyState(
            new_tokens_by_request=new_tokens,
            reused_tokens_by_request=reused_tokens,
            current_layer=int(prefill_batch.split_index),
            total_layers=total_layers,
        ),
        decode=DecodePolicyState(
            context_tokens_by_request=_decode_context(running_batch)
        ),
        current_stream_idx=current_stream_idx,
        trigger=trigger,
    )


__all__ = [
    "DecodePolicyState",
    "PDMuxPolicyInput",
    "PrefillPolicyState",
    "build_policy_input",
]
