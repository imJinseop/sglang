"""CUDA-free data types used by the paper-derived Bullet SRM adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ExecutionState:
    """Bullet's six estimator inputs, plus prefill progress.

    The six estimator dimensions are obtained from these fields and a plan:
    prefill sequence length, prefill batch size, prefill SMs, mean decode
    context length, decode batch size, and decode SMs.
    """

    prefill_tokens: int
    prefill_batch: int
    decode_batch: int
    decode_context_tokens: int
    remaining_layers: int

    def __post_init__(self) -> None:
        if (
            min(
                self.prefill_tokens,
                self.prefill_batch,
                self.decode_batch,
                self.decode_context_tokens,
                self.remaining_layers,
            )
            < 0
        ):
            raise ValueError("execution-state counters must be >= 0")
        if self.remaining_layers == 0:
            raise ValueError("remaining_layers must be > 0")

    @property
    def mean_prefill_sequence(self) -> float:
        return self.prefill_tokens / self.prefill_batch if self.prefill_batch else 0.0

    @property
    def mean_decode_context(self) -> float:
        return (
            self.decode_context_tokens / self.decode_batch if self.decode_batch else 0.0
        )


@dataclass(frozen=True)
class ExecutionPlan:
    prefill_sm: int
    decode_sm: int
    frequency_mhz: int

    def __post_init__(self) -> None:
        if min(self.prefill_sm, self.decode_sm, self.frequency_mhz) <= 0:
            raise ValueError("plan SM counts and frequency must be > 0")

    @property
    def total_sm(self) -> int:
        return self.prefill_sm + self.decode_sm


@dataclass(frozen=True)
class KernelWork:
    name: str
    flops: float
    dram_bytes: float

    def __post_init__(self) -> None:
        if min(self.flops, self.dram_bytes) < 0:
            raise ValueError("kernel work must be >= 0")
        if self.flops == 0 and self.dram_bytes == 0:
            raise ValueError("kernel work cannot be empty")

    def scaled(self, factor: float) -> "KernelWork":
        if factor <= 0:
            raise ValueError("scale factor must be > 0")
        return KernelWork(self.name, self.flops * factor, self.dram_bytes * factor)


@dataclass(frozen=True)
class LaneWork:
    """Ordered kernel work for one scheduling quantum."""

    kernels: tuple[KernelWork, ...]

    def __post_init__(self) -> None:
        if not self.kernels:
            raise ValueError("lane work needs at least one kernel")

    @property
    def flops(self) -> float:
        return sum(kernel.flops for kernel in self.kernels)

    @property
    def dram_bytes(self) -> float:
        return sum(kernel.dram_bytes for kernel in self.kernels)

    @classmethod
    def from_kernels(cls, kernels: Iterable[KernelWork]) -> "LaneWork":
        return cls(tuple(kernels))

    @classmethod
    def from_workload(cls, workload: object) -> "LaneWork":
        operators = getattr(workload, "operators", None)
        if operators is None:
            profile = workload.to_profile()
            return cls(
                (
                    KernelWork(
                        name="aggregate",
                        flops=float(profile.FLOPs),
                        dram_bytes=float(profile.data_bytes),
                    ),
                )
            )
        kernels = tuple(
            KernelWork(
                name=str(getattr(operator, "name", operator.__class__.__name__)),
                flops=float(operator.total_FLOPs),
                dram_bytes=float(operator.total_bytes),
            )
            for operator in operators
            if float(operator.total_FLOPs) > 0 or float(operator.total_bytes) > 0
        )
        return cls(kernels)

    def scaled(self, factor: float) -> "LaneWork":
        return LaneWork(tuple(kernel.scaled(factor) for kernel in self.kernels))


@dataclass(frozen=True)
class CandidateEstimate:
    plan: ExecutionPlan
    decode_latency_s: float
    prefill_latency_s: float

    def __post_init__(self) -> None:
        for latency in (self.decode_latency_s, self.prefill_latency_s):
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("predicted latencies must be finite and > 0")
