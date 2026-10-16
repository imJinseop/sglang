# MI project extension; not part of upstream SGLang.
"""Opt-in ground-truth measurements for real SGLang transformer layers.

The recorder is deliberately dormant unless ``SGLANG_LAYERWISE_GT_PATH`` is
set.  When armed, it repeats selected eager layer calls on their actual SGLang
stream and records CUDA-event latency plus the NVML board-energy delta.  The
same layer inputs are reused for every repetition, so measurement does not turn
the repetitions into additional transformer depth.

CUDA graphs execute without Python module calls.  Ground-truth campaigns must
therefore disable decode CUDA graphs; this is checked explicitly instead of
silently returning capture-time measurements.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, TypeVar

import torch
from sglang.version import __version__ as sglang_version

T = TypeVar("T")


def _selected_layers(raw: str, layer_count: int) -> frozenset[int]:
    if not raw or raw.strip().lower() == "all":
        return frozenset(range(layer_count))
    selected: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"invalid layer range {item!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    invalid = sorted(index for index in selected if not 0 <= index < layer_count)
    if invalid:
        raise ValueError(
            f"ground-truth layer indexes must be in [0, {layer_count - 1}]: {invalid}"
        )
    return frozenset(selected)


def _mode_name(forward_batch) -> str:
    mode = getattr(forward_batch, "forward_mode", None)
    return str(getattr(mode, "name", mode)).lower()


def _phase(forward_batch) -> str:
    name = _mode_name(forward_batch)
    if "decode" in name:
        return "decode"
    if "extend" in name or "prefill" in name:
        return "prefill"
    return name


def _int_tuple(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return tuple(int(item) for item in value)


class LayerwiseGroundTruthRecorder:
    def __init__(self, path: Path, layer_count: int) -> None:
        repetitions = int(os.environ.get("SGLANG_LAYERWISE_GT_REPETITIONS", "16"))
        prefill_repetitions = int(
            os.environ.get("SGLANG_LAYERWISE_GT_PREFILL_REPETITIONS", repetitions)
        )
        decode_repetitions = int(
            os.environ.get("SGLANG_LAYERWISE_GT_DECODE_REPETITIONS", repetitions)
        )
        warmup = int(os.environ.get("SGLANG_LAYERWISE_GT_WARMUP", "3"))
        if min(repetitions, prefill_repetitions, decode_repetitions) <= 0 or warmup < 0:
            raise ValueError(
                "SGLANG layerwise repetition counts must be positive and "
                "SGLANG_LAYERWISE_GT_WARMUP must be nonnegative"
            )
        self.path = path
        self.repetitions_by_phase = {
            "prefill": prefill_repetitions,
            "decode": decode_repetitions,
        }
        self.default_repetitions = repetitions
        self.warmup = warmup
        self.layers = _selected_layers(
            os.environ.get("SGLANG_LAYERWISE_GT_LAYERS", "all"), layer_count
        )
        arm_path = os.environ.get("SGLANG_LAYERWISE_GT_ARM_FILE")
        self.arm_path = Path(arm_path) if arm_path else None
        self._lock = threading.Lock()
        self._forward_index = 0

        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        device_index = torch.cuda.current_device()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self._device_index = device_index

    def armed_for_phase(self, phase: str) -> bool:
        if self.arm_path is None:
            return True
        if not self.arm_path.is_file():
            return False
        control = self.arm_path.read_text(encoding="utf-8").strip().lower()
        if not control or control == "all":
            return True
        return phase in {item.strip() for item in control.split(",")}

    def begin_forward(self) -> int:
        with self._lock:
            value = self._forward_index
            self._forward_index += 1
            return value

    def _energy_mj(self) -> int:
        from utils.observer.queries import read_total_energy_mj

        return int(read_total_energy_mj(self._handle, self._nvml))

    def _clock_mhz(self) -> int:
        return int(
            self._nvml.nvmlDeviceGetClockInfo(self._handle, self._nvml.NVML_CLOCK_SM)
        )

    def _temperature_c(self) -> int:
        return int(
            self._nvml.nvmlDeviceGetTemperature(
                self._handle, self._nvml.NVML_TEMPERATURE_GPU
            )
        )

    def _stream_sm_count(self, stream: torch.cuda.Stream) -> int:
        try:
            from cuda.bindings import driver as cuda_driver

            status, resource = cuda_driver.cuStreamGetDevResource(
                stream.cuda_stream,
                cuda_driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            )
            if status == cuda_driver.CUresult.CUDA_SUCCESS:
                return int(resource.sm.smCount)
        except Exception:
            pass
        return int(
            torch.cuda.get_device_properties(self._device_index).multi_processor_count
        )

    def _metadata(self, forward_batch, hidden_states: torch.Tensor) -> dict:
        context_lengths = _int_tuple(getattr(forward_batch, "seq_lens_cpu", None))
        new_lengths = _int_tuple(getattr(forward_batch, "extend_seq_lens_cpu", None))
        prefix_lengths = _int_tuple(
            getattr(forward_batch, "extend_prefix_lens_cpu", None)
        )
        batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
        if not new_lengths and _phase(forward_batch) == "decode":
            new_lengths = (1,) * batch_size
        return {
            "forward_mode": _mode_name(forward_batch),
            "phase": _phase(forward_batch),
            "batch_size": batch_size,
            "token_count": int(hidden_states.shape[0]),
            "new_tokens_by_request": list(new_lengths),
            "prefix_tokens_by_request": list(prefix_lengths),
            "context_tokens_by_request": list(context_lengths),
            "split_index": int(getattr(forward_batch, "split_index", 0) or 0),
        }

    def measure_layer(
        self,
        *,
        layer_index: int,
        forward_index: int,
        forward_batch,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        call: Callable[[torch.Tensor, torch.Tensor | None], T],
    ) -> T:
        phase = _phase(forward_batch)
        if not self.armed_for_phase(phase) or layer_index not in self.layers:
            return call(hidden_states, residual)
        if os.environ.get('MI_LAYER_COUNTER_RECORDER') == '1':
            from microbench.layer_counters import profile_layer
            return profile_layer(self, layer_index=layer_index, forward_index=forward_index,
                forward_batch=forward_batch, hidden_states=hidden_states, residual=residual, call=call)
        if os.environ.get("MI_UNIT_LAYER_RECORDER") == "1":
            from e2e_inference.isolated_layer_recorder import measure_layer

            return measure_layer(
                self, layer_index=layer_index, forward_index=forward_index,
                forward_batch=forward_batch, hidden_states=hidden_states,
                residual=residual, call=call,
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SGLANG layerwise ground truth cannot run during CUDA graph capture; "
                "set disable_cuda_graph=True"
            )

        # Fused add+RMSNorm may mutate both hidden_states and residual.  Give
        # every repeated layer invocation an independent pre-call snapshot;
        # cloning or resetting inside the timed interval would contaminate the
        # layer latency and energy measurements.
        warmup_inputs = [
            (
                hidden_states.clone(),
                None if residual is None else residual.clone(),
            )
            for _ in range(self.warmup)
        ]
        repetitions = self.repetitions_by_phase.get(phase, self.default_repetitions)
        measured_inputs = [
            (
                hidden_states.clone(),
                None if residual is None else residual.clone(),
            )
            for _ in range(repetitions)
        ]
        for warmup_hidden, warmup_residual in warmup_inputs:
            call(warmup_hidden, warmup_residual)
        stream = torch.cuda.current_stream(self._device_index)
        stream.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        clock_start = self._clock_mhz()
        temperature_start = self._temperature_c()
        energy_start = self._energy_mj()
        start.record(stream)
        for measured_hidden, measured_residual in measured_inputs:
            call(measured_hidden, measured_residual)
        end.record(stream)
        end.synchronize()
        energy_end = self._energy_mj()
        clock_end = self._clock_mhz()
        temperature_end = self._temperature_c()
        elapsed_ms = float(start.elapsed_time(end))
        energy_mj = float(energy_end - energy_start)
        if energy_mj < 0:
            raise RuntimeError("NVML total-energy counter moved backwards")

        row = {
            "schema_version": 1,
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "sglang_version": sglang_version,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(self._device_index),
            "device_index": self._device_index,
            "forward_index": forward_index,
            "layer_index": layer_index,
            "repetitions": repetitions,
            "warmup_repetitions": self.warmup,
            "elapsed_ms": elapsed_ms,
            "latency_ms": elapsed_ms / repetitions,
            "energy_mJ": energy_mj / repetitions,
            "measured_total_energy_mJ": energy_mj,
            "sm_count": self._stream_sm_count(stream),
            "sm_clock_mhz_start": clock_start,
            "sm_clock_mhz_end": clock_end,
            "sm_clock_mhz": (clock_start + clock_end) / 2.0,
            "temperature_c_start": temperature_start,
            "temperature_c_end": temperature_end,
            **self._metadata(forward_batch, hidden_states),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")
        # The snapshots above exist only for measurement.  Execute the real
        # model call once on its original inputs so downstream layers observe
        # exactly the uninstrumented forward semantics.
        return call(hidden_states, residual)


_RECORDER: LayerwiseGroundTruthRecorder | None | bool = False


def get_layerwise_ground_truth_recorder(
    layer_count: int,
) -> LayerwiseGroundTruthRecorder | None:
    global _RECORDER
    if _RECORDER is False:
        raw_path = os.environ.get("SGLANG_LAYERWISE_GT_PATH")
        _RECORDER = (
            LayerwiseGroundTruthRecorder(Path(raw_path), layer_count)
            if raw_path
            else None
        )
    return _RECORDER if isinstance(_RECORDER, LayerwiseGroundTruthRecorder) else None


__all__ = [
    "LayerwiseGroundTruthRecorder",
    "get_layerwise_ground_truth_recorder",
]
