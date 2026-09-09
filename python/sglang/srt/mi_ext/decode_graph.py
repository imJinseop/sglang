# MI project extension; not part of upstream SGLang.
"""Decode graph selection and experiment telemetry for PDMux/OmniDecode."""

from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    get_sm_counts,
    get_stream_groups,
)
from .multiplex.context import read_decode_graph_mode


def init_general_graph_state(self):
    full_sm_once = (
        self.enable_pdmux
        and read_decode_graph_mode(self.model_runner.server_args.pdmux_config_path)
        == "full_sm_once"
    )
    if not full_sm_once:
        self.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)


def capture_statistics(self) -> dict[str, object]:
    """Return the realized decode-graph capture cardinality.

    ``capture_bs`` alone is insufficient for PDMux because the default
    mode captures every batch bucket once per green-context stream.  The
    backend dictionary is populated after capture and therefore reports
    the actual number of replayable graph artifacts, including optional
    LoRA/DSA variants.
    """
    captured_graphs = getattr(self.backend, "_graphs", {})
    capture_stream_count = (
        1
        if self.enable_pdmux and self.pdmux_decode_graph_mode == "full_sm_once"
        else len(self.stream_groups) if self.enable_pdmux else 1
    )
    return {
        "mode": (self.pdmux_decode_graph_mode if self.enable_pdmux else "standard"),
        "batch_sizes": list(self.capture_bs),
        "batch_size_count": len(self.capture_bs),
        "capture_stream_count": capture_stream_count,
        "graph_artifact_count": len(captured_graphs),
    }


def maybe_init_pdmux(self):
    if self.enable_pdmux:
        self.stream_groups = get_stream_groups()
        self.pdmux_decode_graph_mode = read_decode_graph_mode(
            self.model_runner.server_args.pdmux_config_path
        )
        sm_counts = get_sm_counts()
        self.pdmux_decode_capture_stream_idx = max(
            range(len(sm_counts)), key=lambda index: sm_counts[index][1]
        )
        graph_backends = (
            (
                self.model_runner.decode_attn_backend_group[
                    self.pdmux_decode_capture_stream_idx
                ],
            )
            if self.pdmux_decode_graph_mode == "full_sm_once"
            else self.model_runner.decode_attn_backend_group
        )
        for attn_backend in graph_backends:
            attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)


def pdmux_graph_stream_idx(self) -> int:
    if getattr(self, "pdmux_decode_graph_mode", "per_partition") == "full_sm_once":
        return self.pdmux_decode_capture_stream_idx
    return get_current_stream_idx()


def graph_capture_statistics(self) -> dict:
    """Aggregate realized decode CUDA-graph counts across model runners."""
    runners = self.model_runner_list or [self.model_runner]
    per_runner = []
    for runner in runners:
        decode_runner = getattr(runner, "decode_cuda_graph_runner", None)
        if decode_runner is None or not hasattr(decode_runner, "capture_bs"):
            continue
        per_runner.append(capture_statistics(decode_runner))
    return {
        "decode_graph_artifact_count": sum(
            int(item["graph_artifact_count"]) for item in per_runner
        ),
        "decode_capture_stream_count": sum(
            int(item["capture_stream_count"]) for item in per_runner
        ),
        "decode_capture_batch_size_count": sum(
            int(item["batch_size_count"]) for item in per_runner
        ),
        "model_runners": per_runner,
    }


def capture_stream_groups(self):
    if self.pdmux_decode_graph_mode == "full_sm_once":
        index = self.pdmux_decode_capture_stream_idx
        return ((index, self.stream_groups[index]),)
    return tuple(enumerate(self.stream_groups))
