# MI project extension; not part of upstream SGLang.
"""Independent implementation of Bullet's SM-scaling roofline model.

The public BulletServe repository loads a private native predictor library; it
does not contain the predictor implementation or calibrated parameters. This
module therefore implements only what the paper specifies and keeps every
profile value serializable. It is not the authors' missing predictor.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from .types import CandidateEstimate, ExecutionPlan, ExecutionState, LaneWork

Phase = Literal["decode", "prefill"]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BandwidthContentionPoint:
    """Measured decode-bandwidth multiplier for a prefill sequence length."""

    prefill_sequence_tokens: float
    bandwidth_multiplier: float
    decode_sm: int | None = None

    def __post_init__(self) -> None:
        if self.prefill_sequence_tokens < 0:
            raise ValueError("prefill_sequence_tokens must be >= 0")
        if not 0 < self.bandwidth_multiplier <= 1:
            raise ValueError("bandwidth_multiplier must be in (0, 1]")
        if self.decode_sm is not None and self.decode_sm <= 0:
            raise ValueError("decode_sm must be positive")


@dataclass(frozen=True)
class SRMHardwareProfile:
    """The measured capacity terms in Bullet Equation 1."""

    device_name: str
    total_sm: int
    reference_frequency_mhz: int
    peak_compute_flops_s: float
    peak_memory_bandwidth_Bps: float
    memory_saturation_sm: float
    decode_contention: tuple[BandwidthContentionPoint, ...] = ()
    provenance: str = "user-supplied"

    def __post_init__(self) -> None:
        if self.total_sm <= 1 or self.reference_frequency_mhz <= 0:
            raise ValueError("total_sm and frequency must be positive")
        if (
            min(
                self.peak_compute_flops_s,
                self.peak_memory_bandwidth_Bps,
                self.memory_saturation_sm,
            )
            <= 0
        ):
            raise ValueError("SRM hardware rates must be > 0")
        if self.memory_saturation_sm > self.total_sm:
            raise ValueError("memory_saturation_sm cannot exceed total_sm")
        lengths = tuple((point.decode_sm or 0, point.prefill_sequence_tokens)
            for point in self.decode_contention)
        if lengths != tuple(sorted(set(lengths))):
            raise ValueError("decode_contention must be unique and sorted")

    @classmethod
    def from_device_name(
        cls,
        device_name: str,
        *,
        dtype: str,
        reference_frequency_mhz: int,
        device_dir: str | None = None,
    ) -> "SRMHardwareProfile":
        """Bootstrap capacity terms from MI's measured device curves.

        This is convenient for a smoke test, but a Bullet comparison still
        needs sparse end-to-end calibration and contention measurements.
        """

        from device import Device
        from modeling.cost_model.latency.model import (
            bandwidth_curve_from_device, compute_curve_from_device, compute_rate,
            saturation_bandwidth_Bps,
        )

        mode = {
            "bf16": "bf16_tensor",
            "fp16": "fp16_tensor",
            "float16": "fp16_tensor",
            "tf32": "tf32_tensor",
            "fp32": "fp32",
            "int8": "int8_tensor",
        }.get(dtype.lower())
        if mode is None:
            raise ValueError(f"unsupported dtype {dtype!r}")
        device = Device.load(device_name, device_dir)
        compute = compute_curve_from_device(device, mode)
        bandwidth = bandwidth_curve_from_device(device, "memcpy")
        frequency = int(reference_frequency_mhz)
        peak_bandwidth = saturation_bandwidth_Bps(bandwidth)
        return cls(
            device_name=device.name,
            total_sm=int(device.specification.num_sm),
            reference_frequency_mhz=frequency,
            peak_compute_flops_s=compute_rate(
                compute, frequency, device.specification.num_sm
            ),
            peak_memory_bandwidth_Bps=peak_bandwidth,
            memory_saturation_sm=min(
                float(device.specification.num_sm),
                peak_bandwidth / (bandwidth.slope_Bps_per_MHz_SM * frequency),
            ),
            provenance="MI measured capacity curves; Bullet Eq. 1 reproduction",
        )

    def decode_bandwidth_multiplier(self, prefill_tokens: float, decode_sm: int | None = None) -> float:
        points = self.decode_contention
        if any(point.decode_sm is not None for point in points):
            matches = tuple(point for point in points if point.decode_sm == decode_sm)
            if not matches:
                raise ValueError(f"no measured Bullet contention curve for decode_sm={decode_sm}")
            points = matches
        if not points:
            return 1.0
        if prefill_tokens <= points[0].prefill_sequence_tokens:
            return points[0].bandwidth_multiplier
        for left, right in zip(points, points[1:]):
            if prefill_tokens <= right.prefill_sequence_tokens:
                fraction = (prefill_tokens - left.prefill_sequence_tokens) / (
                    right.prefill_sequence_tokens - left.prefill_sequence_tokens
                )
                return left.bandwidth_multiplier + fraction * (
                    right.bandwidth_multiplier - left.bandwidth_multiplier
                )
        return points[-1].bandwidth_multiplier

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> "SRMHardwareProfile":
        return cls(
            device_name=str(raw["device_name"]),
            total_sm=int(raw["total_sm"]),
            reference_frequency_mhz=int(raw["reference_frequency_mhz"]),
            peak_compute_flops_s=float(raw["peak_compute_flops_s"]),
            peak_memory_bandwidth_Bps=float(raw["peak_memory_bandwidth_Bps"]),
            memory_saturation_sm=float(raw["memory_saturation_sm"]),
            decode_contention=tuple(
                BandwidthContentionPoint(**point)  # type: ignore[arg-type]
                for point in raw.get("decode_contention", ())  # type: ignore[union-attr]
            ),
            provenance=str(raw.get("provenance", "user-supplied")),
        )


def execution_features(
    state: ExecutionState, plan: ExecutionPlan
) -> tuple[float, float, float, float, float, float]:
    return (
        state.mean_prefill_sequence,
        float(state.prefill_batch),
        float(plan.prefill_sm),
        state.mean_decode_context,
        float(state.decode_batch),
        float(plan.decode_sm),
    )


@dataclass(frozen=True)
class SRMCalibrationPoint:
    """One measured alpha=T_measured/T_roofline at an execution state."""

    phase: Phase
    features: tuple[float, float, float, float, float, float]
    scaling_factor: float
    source: str = "offline"

    def __post_init__(self) -> None:
        if self.phase not in {"decode", "prefill"}:
            raise ValueError("invalid phase")
        if len(self.features) != 6 or self.scaling_factor <= 0:
            raise ValueError("calibration needs six features and alpha > 0")

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> "SRMCalibrationPoint":
        features = tuple(float(value) for value in raw["features"])  # type: ignore[arg-type]
        if len(features) != 6:
            raise ValueError("calibration features must have length six")
        return cls(
            phase=str(raw["phase"]),  # type: ignore[arg-type]
            features=features,  # type: ignore[arg-type]
            scaling_factor=float(raw["scaling_factor"]),
            source=str(raw.get("source", "offline")),
        )


@dataclass(frozen=True)
class SRMCalibration:
    points: tuple[SRMCalibrationPoint, ...] = ()
    neighbor_count: int = 12
    ridge: float = 1e-3
    max_extrapolation_ratio: float = 2.0
    metadata: Mapping[str, str] = field(default_factory=dict)
    _phase_arrays: Mapping[
        Phase, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        cache = {}
        for phase in ("decode", "prefill"):
            points = tuple(point for point in self.points if point.phase == phase)
            if not points:
                continue
            x = np.asarray([point.features for point in points], dtype=np.float64)
            y = np.asarray([point.scaling_factor for point in points], dtype=np.float64)
            spans = np.ptp(x, axis=0)
            active = spans > 0
            # Profile arrays are immutable after construction. Rebuilding them
            # at every candidate prediction dominated the Python adapter's
            # scheduling time by three orders of magnitude over the paper's
            # native predictor.
            for array in (x, y, spans, active):
                array.setflags(write=False)
            cache[phase] = (x, y, spans, active)
        object.__setattr__(self, "_phase_arrays", cache)

    @property
    def calibrated(self) -> bool:
        return bool(self.points)

    def scaling_factor(
        self, phase: Phase, state: ExecutionState, plan: ExecutionPlan
    ) -> float:
        """Normalized local linear alpha interpolation and extrapolation.

        Bullet specifies linear extrapolation without its exact estimator. Fit
        a tiny weighted affine model around the query; ridge regularizes slopes
        only. An exact measured state returns its median alpha. Positivity and
        extrapolation bounds are explicit reproduction parameters.
        """

        cached = self._phase_arrays.get(phase)
        if cached is None:
            return 1.0
        query = np.asarray(execution_features(state, plan), dtype=np.float64)
        x, y, spans, active = cached
        exact = np.all(np.isclose(x, query, rtol=0, atol=1e-12), axis=1)
        if np.any(exact):
            return float(np.median(y[exact]))
        delta = (x - query) / np.where(active, spans, 1.0)
        delta[:, ~active] = 0
        distance = np.linalg.norm(delta, axis=1)
        count = min(self.neighbor_count, len(distance))
        nearest = np.argpartition(distance, count - 1)[:count]
        values = y[nearest]
        weights = 1 / np.maximum(distance[nearest], 1e-6)
        dx = delta[nearest][:, active]
        design = np.column_stack((np.ones(count), dx))
        regularizer = np.eye(design.shape[1]) * self.ridge
        regularizer[0, 0] = 0.0
        try:
            coefficients = np.linalg.solve(
                design.T @ (weights[:, None] * design) + regularizer,
                design.T @ (weights * values),
            )
            predicted = float(coefficients[0])
        except np.linalg.LinAlgError:
            predicted = float(np.average(values, weights=weights))
        return max(
            float(values.min()) / self.max_extrapolation_ratio,
            min(float(values.max()) * self.max_extrapolation_ratio, predicted),
        )

    def refined(
        self,
        point: SRMCalibrationPoint,
        *,
        online_weight: float,
    ) -> "SRMCalibration":
        if not 0 < online_weight <= 1:
            raise ValueError("online_weight must be in (0, 1]")
        points = list(self.points)
        for index, old in enumerate(points):
            if old.phase == point.phase and old.features == point.features:
                points[index] = replace(
                    point,
                    scaling_factor=(
                        (1 - online_weight) * old.scaling_factor
                        + online_weight * point.scaling_factor
                    ),
                    source="online-ema",
                )
                break
        else:
            points.append(point)
        return replace(self, points=tuple(points))


class BulletSRMPredictor:
    """Kernel SRM -> phase quantum -> scheduling-candidate latency."""

    def __init__(
        self,
        hardware: SRMHardwareProfile,
        calibration: SRMCalibration | None = None,
    ) -> None:
        self.hardware = hardware
        self.calibration = calibration or SRMCalibration()

    @property
    def total_sm(self) -> int:
        return self.hardware.total_sm

    @property
    def reference_frequency_mhz(self) -> int:
        return self.hardware.reference_frequency_mhz

    @property
    def calibrated(self) -> bool:
        return self.calibration.calibrated

    def _capacities(
        self, phase: Phase, state: ExecutionState, plan: ExecutionPlan
    ) -> tuple[float, float]:
        if plan.total_sm != self.total_sm:
            raise ValueError("plan and hardware total SM counts differ")
        if plan.frequency_mhz != self.reference_frequency_mhz:
            raise ValueError("this SRM profile is valid at one frequency only")
        sm = plan.decode_sm if phase == "decode" else plan.prefill_sm
        compute = self.hardware.peak_compute_flops_s * sm / self.total_sm
        bandwidth = self.hardware.peak_memory_bandwidth_Bps * min(
            1.0, sm / self.hardware.memory_saturation_sm
        )
        if phase == "decode" and state.prefill_tokens:
            bandwidth *= self.hardware.decode_bandwidth_multiplier(
                state.mean_prefill_sequence, plan.decode_sm
            )
        return compute, bandwidth

    def roofline_latency(
        self,
        phase: Phase,
        state: ExecutionState,
        work: LaneWork,
        plan: ExecutionPlan,
    ) -> float:
        compute, bandwidth = self._capacities(phase, state, plan)
        # Bullet applies Eq. 1 to every kernel and then sums kernel latency.
        return sum(
            max(kernel.flops / compute, kernel.dram_bytes / bandwidth)
            for kernel in work.kernels
        )

    def phase_latency(
        self,
        phase: Phase,
        state: ExecutionState,
        work: LaneWork,
        plan: ExecutionPlan,
    ) -> float:
        return self.roofline_latency(phase, state, work, plan) * (
            self.calibration.scaling_factor(phase, state, plan)
        )

    def predict(
        self,
        state: ExecutionState,
        decode_work: LaneWork,
        prefill_work: LaneWork,
        plan: ExecutionPlan,
    ) -> CandidateEstimate:
        return CandidateEstimate(
            plan=plan,
            decode_latency_s=self.phase_latency("decode", state, decode_work, plan),
            prefill_latency_s=self.phase_latency("prefill", state, prefill_work, plan),
        )

    def with_online_observation(
        self,
        *,
        phase: Phase,
        state: ExecutionState,
        work: LaneWork,
        plan: ExecutionPlan,
        measured_latency_s: float,
        online_weight: float = 0.25,
    ) -> "BulletSRMPredictor":
        if not math.isfinite(measured_latency_s) or measured_latency_s <= 0:
            raise ValueError("measured latency must be finite and > 0")
        roofline = self.roofline_latency(phase, state, work, plan)
        point = SRMCalibrationPoint(
            phase=phase,
            features=execution_features(state, plan),
            scaling_factor=measured_latency_s / roofline,
            source="online",
        )
        return BulletSRMPredictor(
            self.hardware,
            self.calibration.refined(point, online_weight=online_weight),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": "bullet_srm_independent_reproduction",
            "hardware": self.hardware.to_json(),
            "calibration": {
                "points": [asdict(point) for point in self.calibration.points],
                "neighbor_count": self.calibration.neighbor_count,
                "ridge": self.calibration.ridge,
                "max_extrapolation_ratio": (self.calibration.max_extrapolation_ratio),
                "metadata": dict(self.calibration.metadata),
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BulletSRMPredictor":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("unsupported Bullet SRM profile schema")
        calibration_raw = raw.get("calibration") or {}
        calibration = SRMCalibration(
            points=tuple(
                SRMCalibrationPoint.from_json(point)
                for point in calibration_raw.get("points", ())
            ),
            neighbor_count=int(calibration_raw.get("neighbor_count", 12)),
            ridge=float(calibration_raw.get("ridge", 1e-3)),
            max_extrapolation_ratio=float(
                calibration_raw.get("max_extrapolation_ratio", 2.0)
            ),
            metadata={
                str(key): str(value)
                for key, value in (calibration_raw.get("metadata") or {}).items()
            },
        )
        return cls(SRMHardwareProfile.from_json(raw["hardware"]), calibration)
