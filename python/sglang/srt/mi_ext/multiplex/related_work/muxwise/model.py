# MI project extension; not part of upstream SGLang.
"""Inspectable reproduction of MuxWise's published latency estimator.

The authors' public artifact does not contain this model.  This module follows
MuxWise Equations 1 and 2 and its grid-cell maximum decode slowdown guard.  It
does not import Torch or SGLang, so profiles and decisions can be tested on CPU.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

Phase = Literal["prefill", "decode"]
SCHEMA_VERSION = 1


def _nonnegative_ints(name: str, values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} values must be >= 0")
    return result


@dataclass(frozen=True)
class PhaseCoefficients:
    """One phase's solo-run regression at one fixed SM count.

    Prefill theta has four entries for ``sum(n^2), sum(n*r), sum(n), 1``.
    Decode theta has three entries for ``sum(r), batch_size, 1``.
    Coefficients predict the latency of a full-model phase in milliseconds.
    """

    phase: Phase
    sm: int
    theta: tuple[float, ...]
    fit_samples: int = 0
    fit_mape: float | None = None

    def __post_init__(self) -> None:
        expected = 4 if self.phase == "prefill" else 3
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be 'prefill' or 'decode'")
        if self.sm <= 0:
            raise ValueError("sm must be > 0")
        if len(self.theta) != expected:
            raise ValueError(f"{self.phase} theta must contain {expected} coefficients")
        if not all(math.isfinite(value) for value in self.theta):
            raise ValueError("theta coefficients must be finite")
        if self.fit_samples < 0:
            raise ValueError("fit_samples must be >= 0")
        if self.fit_mape is not None and self.fit_mape < 0:
            raise ValueError("fit_mape must be >= 0")

    def predict_prefill_ms(
        self,
        new_tokens: Sequence[int],
        reused_tokens: Sequence[int],
    ) -> float:
        if self.phase != "prefill":
            raise ValueError("decode coefficients cannot predict prefill")
        new = _nonnegative_ints("new_tokens", new_tokens)
        reused = _nonnegative_ints("reused_tokens", reused_tokens)
        if not new or len(new) != len(reused):
            raise ValueError("prefill new/reused vectors must be non-empty and aligned")
        features = (
            float(sum(value * value for value in new)),
            float(sum(n * r for n, r in zip(new, reused))),
            float(sum(new)),
            1.0,
        )
        return self._positive_prediction(features)

    def predict_decode_ms(self, context_tokens: Sequence[int]) -> float:
        if self.phase != "decode":
            raise ValueError("prefill coefficients cannot predict decode")
        context = _nonnegative_ints("context_tokens", context_tokens)
        if not context:
            raise ValueError("decode context vector must be non-empty")
        features = (float(sum(context)), float(len(context)), 1.0)
        return self._positive_prediction(features)

    def _positive_prediction(self, features: Sequence[float]) -> float:
        prediction = sum(a * b for a, b in zip(self.theta, features))
        if not math.isfinite(prediction) or prediction <= 0:
            raise ValueError(
                f"{self.phase} regression produced invalid latency {prediction}; "
                "the profile needs more or better-conditioned samples"
            )
        return prediction

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["theta"] = list(self.theta)
        return result

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> "PhaseCoefficients":
        return cls(
            phase=str(raw["phase"]),  # type: ignore[arg-type]
            sm=int(raw["sm"]),
            theta=tuple(float(value) for value in raw["theta"]),  # type: ignore[arg-type]
            fit_samples=int(raw.get("fit_samples", 0)),
            fit_mape=(None if raw.get("fit_mape") is None else float(raw["fit_mape"])),
        )


@dataclass(frozen=True)
class GuardCell:
    """Worst observed decode slowdown in one MuxWise grid cell."""

    prefill_sm: int
    decode_sm: int
    prefill_new_bucket: int
    prefill_reused_bucket: int
    decode_batch_bucket: int
    decode_reused_bucket: int
    max_slowdown: float
    samples: int = 1

    def __post_init__(self) -> None:
        if min(self.prefill_sm, self.decode_sm) <= 0:
            raise ValueError("guard partitions must be > 0")
        if (
            min(
                self.prefill_new_bucket,
                self.prefill_reused_bucket,
                self.decode_batch_bucket,
                self.decode_reused_bucket,
            )
            < 0
        ):
            raise ValueError("guard bucket indices must be >= 0")
        if not math.isfinite(self.max_slowdown) or self.max_slowdown < 1:
            raise ValueError("max_slowdown must be finite and >= 1")
        if self.samples <= 0:
            raise ValueError("guard cell samples must be > 0")

    @property
    def key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.prefill_sm,
            self.decode_sm,
            self.prefill_new_bucket,
            self.prefill_reused_bucket,
            self.decode_batch_bucket,
            self.decode_reused_bucket,
        )


@dataclass(frozen=True)
class ContentionGuard:
    """Five-dimensional grid and maximum slowdown lookup for decode."""

    prefill_new_boundaries: tuple[int, ...]
    prefill_reused_boundaries: tuple[int, ...]
    decode_batch_boundaries: tuple[int, ...]
    decode_reused_boundaries: tuple[int, ...]
    cells: tuple[GuardCell, ...]
    default_max_slowdown: float

    def __post_init__(self) -> None:
        for name in (
            "prefill_new_boundaries",
            "prefill_reused_boundaries",
            "decode_batch_boundaries",
            "decode_reused_boundaries",
        ):
            values = getattr(self, name)
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be non-empty, unique, and sorted")
            if values[0] <= 0:
                raise ValueError(f"{name} values must be > 0")
        if self.default_max_slowdown < 1:
            raise ValueError("default_max_slowdown must be >= 1")
        keys = tuple(cell.key for cell in self.cells)
        if len(keys) != len(set(keys)):
            raise ValueError("contention guard contains duplicate cells")

    @staticmethod
    def _bucket(boundaries: tuple[int, ...], value: int) -> int:
        return bisect.bisect_left(boundaries, max(0, int(value)))

    def key(
        self,
        *,
        prefill_sm: int,
        decode_sm: int,
        prefill_new_tokens: int,
        prefill_reused_tokens: int,
        decode_batch: int,
        decode_reused_tokens: int,
    ) -> tuple[int, int, int, int, int, int]:
        return (
            int(prefill_sm),
            int(decode_sm),
            self._bucket(self.prefill_new_boundaries, prefill_new_tokens),
            self._bucket(self.prefill_reused_boundaries, prefill_reused_tokens),
            self._bucket(self.decode_batch_boundaries, decode_batch),
            self._bucket(self.decode_reused_boundaries, decode_reused_tokens),
        )

    def lookup(self, **state: int) -> float:
        key = self.key(**state)
        for cell in self.cells:
            if cell.key == key:
                return cell.max_slowdown
        same_plan = tuple(
            cell.max_slowdown
            for cell in self.cells
            if cell.prefill_sm == key[0] and cell.decode_sm == key[1]
        )
        # An unseen cell must not become optimistic.  Use the worst observation
        # for this partition, then the fitted profile's global fallback.
        return max(same_plan, default=self.default_max_slowdown)

    def refined(self, *, observed_slowdown: float, **state: int) -> "ContentionGuard":
        """Monotonically update one cell with a runtime decode observation."""

        if not math.isfinite(observed_slowdown) or observed_slowdown <= 0:
            raise ValueError("observed_slowdown must be finite and > 0")
        slowdown = max(1.0, observed_slowdown)
        key = self.key(**state)
        cells = list(self.cells)
        for index, cell in enumerate(cells):
            if cell.key == key:
                cells[index] = replace(
                    cell,
                    max_slowdown=max(cell.max_slowdown, slowdown),
                    samples=cell.samples + 1,
                )
                break
        else:
            cells.append(
                GuardCell(
                    prefill_sm=key[0],
                    decode_sm=key[1],
                    prefill_new_bucket=key[2],
                    prefill_reused_bucket=key[3],
                    decode_batch_bucket=key[4],
                    decode_reused_bucket=key[5],
                    max_slowdown=slowdown,
                )
            )
        return replace(self, cells=tuple(cells))

    def to_json(self) -> dict[str, object]:
        return {
            "prefill_new_boundaries": list(self.prefill_new_boundaries),
            "prefill_reused_boundaries": list(self.prefill_reused_boundaries),
            "decode_batch_boundaries": list(self.decode_batch_boundaries),
            "decode_reused_boundaries": list(self.decode_reused_boundaries),
            "default_max_slowdown": self.default_max_slowdown,
            "cells": [asdict(cell) for cell in self.cells],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> "ContentionGuard":
        return cls(
            prefill_new_boundaries=tuple(
                int(value) for value in raw["prefill_new_boundaries"]  # type: ignore[arg-type]
            ),
            prefill_reused_boundaries=tuple(
                int(value) for value in raw["prefill_reused_boundaries"]  # type: ignore[arg-type]
            ),
            decode_batch_boundaries=tuple(
                int(value) for value in raw["decode_batch_boundaries"]  # type: ignore[arg-type]
            ),
            decode_reused_boundaries=tuple(
                int(value) for value in raw["decode_reused_boundaries"]  # type: ignore[arg-type]
            ),
            default_max_slowdown=float(raw["default_max_slowdown"]),
            cells=tuple(
                GuardCell(**cell)  # type: ignore[arg-type]
                for cell in raw.get("cells", ())  # type: ignore[union-attr]
            ),
        )


@dataclass(frozen=True)
class MuxWiseEstimate:
    prefill_solo_ms: float
    decode_solo_ms: float
    decode_guard: float
    decode_worst_case_ms: float


@dataclass(frozen=True)
class MuxWiseProfile:
    """Complete per-model/per-machine MuxWise estimator profile."""

    model: str
    device: str
    total_sm: int
    prefill_models: tuple[PhaseCoefficients, ...]
    decode_models: tuple[PhaseCoefficients, ...]
    contention_guard: ContentionGuard
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.model or not self.device:
            raise ValueError("model and device names must not be empty")
        if self.total_sm <= 1:
            raise ValueError("total_sm must be > 1")
        if not self.prefill_models or not self.decode_models:
            raise ValueError("both phase regressions are required")
        if any(item.phase != "prefill" for item in self.prefill_models):
            raise ValueError("prefill_models contains a non-prefill regression")
        if any(item.phase != "decode" for item in self.decode_models):
            raise ValueError("decode_models contains a non-decode regression")
        for models in (self.prefill_models, self.decode_models):
            sms = tuple(item.sm for item in models)
            if len(sms) != len(set(sms)):
                raise ValueError("duplicate phase regression for one SM count")

    @staticmethod
    def _for_sm(
        models: Sequence[PhaseCoefficients],
        sm: int,
        *,
        allow_interpolation: bool = False,
    ) -> PhaseCoefficients:
        for model in models:
            if model.sm == sm:
                return model
        ordered = sorted(models, key=lambda item: item.sm)
        if allow_interpolation:
            for lower, upper in zip(ordered, ordered[1:]):
                if lower.sm < sm < upper.sm:
                    fraction = (sm - lower.sm) / (upper.sm - lower.sm)
                    theta = tuple(
                        left + (right - left) * fraction
                        for left, right in zip(lower.theta, upper.theta)
                    )
                    return PhaseCoefficients(
                        phase=lower.phase,
                        sm=sm,
                        theta=theta,
                        fit_samples=0,
                        fit_mape=None,
                    )
        raise ValueError(
            f"profile has no regression for sm={sm}; profile every pre-created "
            "candidate or enable interpolation between measured SM counts"
        )

    def estimate(
        self,
        *,
        prefill_sm: int,
        decode_sm: int,
        prefill_new_tokens: Sequence[int],
        prefill_reused_tokens: Sequence[int],
        decode_context_tokens: Sequence[int],
        allow_sm_interpolation: bool = False,
    ) -> MuxWiseEstimate:
        if prefill_sm + decode_sm != self.total_sm:
            raise ValueError("candidate SM counts do not match the fitted GPU")
        prefill_model = self._for_sm(
            self.prefill_models,
            prefill_sm,
            allow_interpolation=allow_sm_interpolation,
        )
        decode_model = self._for_sm(
            self.decode_models,
            decode_sm,
            allow_interpolation=allow_sm_interpolation,
        )
        prefill_ms = prefill_model.predict_prefill_ms(
            prefill_new_tokens, prefill_reused_tokens
        )
        decode_ms = decode_model.predict_decode_ms(decode_context_tokens)
        guard = self.contention_guard.lookup(
            prefill_sm=prefill_sm,
            decode_sm=decode_sm,
            prefill_new_tokens=sum(prefill_new_tokens),
            prefill_reused_tokens=sum(prefill_reused_tokens),
            decode_batch=len(decode_context_tokens),
            decode_reused_tokens=sum(decode_context_tokens),
        )
        return MuxWiseEstimate(
            prefill_solo_ms=prefill_ms,
            decode_solo_ms=decode_ms,
            decode_guard=guard,
            decode_worst_case_ms=decode_ms * guard,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "device": self.device,
            "total_sm": self.total_sm,
            "prefill_models": [item.to_json() for item in self.prefill_models],
            "decode_models": [item.to_json() for item in self.decode_models],
            "contention_guard": self.contention_guard.to_json(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> "MuxWiseProfile":
        version = int(raw.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported MuxWise profile schema {version}; expected {SCHEMA_VERSION}"
            )
        return cls(
            model=str(raw["model"]),
            device=str(raw["device"]),
            total_sm=int(raw["total_sm"]),
            prefill_models=tuple(
                PhaseCoefficients.from_json(item)
                for item in raw["prefill_models"]  # type: ignore[arg-type]
            ),
            decode_models=tuple(
                PhaseCoefficients.from_json(item)
                for item in raw["decode_models"]  # type: ignore[arg-type]
            ),
            contention_guard=ContentionGuard.from_json(
                raw["contention_guard"]  # type: ignore[arg-type]
            ),
            metadata={
                str(key): str(value)
                for key, value in (raw.get("metadata") or {}).items()  # type: ignore[union-attr]
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "MuxWiseProfile":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("MuxWise profile must contain a JSON object")
        return cls.from_json(raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), indent=2) + "\n",
            encoding="utf-8",
        )
