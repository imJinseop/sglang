"""CPU-only contracts for keeping native and adaptive PDMux isolated."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mi_ext.multiplex.adaptive_pdmux_config import (
    AdaptivePDMuxConfig,
    load_adaptive_pdmux_config,
)
from sglang.srt.mi_ext.multiplex.context import (apply_concurrency_mode, decode_execution_stream_idx, read_decode_graph_mode)
from sglang.srt.multiplex.pdmux_context import (load_pdmux_config)
from sglang.srt.mi_ext.multiplex.multiplexing_dispatch_mixin import (
    SchedulerMultiplexDispatchMixin,
    SchedulerNativeMultiplexMixin,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _scheduler():
    scheduler = object.__new__(SchedulerMultiplexDispatchMixin)
    scheduler.ps = SimpleNamespace(gpu_id=3, tp_size=2)
    scheduler.init_adaptive_pdmux = MagicMock()
    return scheduler


class TestPDMuxRuntimeDispatch(CustomTestCase):
    def test_single_graph_mode_uses_full_decode_backend_for_eager_fallback(self):
        sm_counts = [(108, 0), (80, 28), (64, 44), (0, 108)]
        config = SimpleNamespace(decode_graph_mode="full_sm_once")
        self.assertEqual(decode_execution_stream_idx(config, sm_counts, 1), 3)
        config.decode_graph_mode = "per_partition"
        self.assertEqual(decode_execution_stream_idx(config, sm_counts, 1), 1)

    def test_decode_capture_statistics_report_realized_graph_cardinality(self):
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.backend = SimpleNamespace(
            _graphs={index: object() for index in range(4)}
        )
        runner.enable_pdmux = True
        runner.pdmux_decode_graph_mode = "full_sm_once"
        runner.stream_groups = [(None, None)] * 8
        runner.capture_bs = [1, 2, 4, 8]

        from sglang.srt.mi_ext.decode_graph import capture_statistics
        stats = capture_statistics(runner)

        self.assertEqual(stats["graph_artifact_count"], 4)
        self.assertEqual(stats["capture_stream_count"], 1)
        self.assertEqual(stats["batch_size_count"], 4)

    def test_single_graph_mode_initializes_only_full_decode_graph_backend(self):
        runner = object.__new__(DecodeCudaGraphRunner)
        backends = [MagicMock() for _ in range(4)]
        runner.enable_pdmux = True
        runner.max_bs = 32
        runner.max_num_token = 32
        runner.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(pdmux_config_path="pdmux.yaml"),
            decode_attn_backend_group=backends,
        )
        stream_groups = [(object(), object()) for _ in range(4)]
        with (
            patch(
                "sglang.srt.mi_ext.decode_graph.get_stream_groups",
                return_value=stream_groups,
            ),
            patch(
                "sglang.srt.mi_ext.decode_graph.get_sm_counts",
                return_value=[(108, 0), (80, 28), (64, 44), (0, 108)],
            ),
            patch(
                "sglang.srt.mi_ext.decode_graph.read_decode_graph_mode",
                return_value="full_sm_once",
            ),
        ):
            runner.maybe_init_pdmux()

        for backend in backends[:-1]:
            backend.init_cuda_graph_state.assert_not_called()
        backends[-1].init_cuda_graph_state.assert_called_once_with(32, 32)

    def test_fixed_partition_concurrency_override_replaces_candidate_menu(self):
        with patch.dict(
            "os.environ",
            {
                "SGLANG_PDMUX_CONCURRENCY_MODE": "fixed_partition",
                "SGLANG_PDMUX_FIXED_DIVISION": "80,28",
            },
        ):
            raw = apply_concurrency_mode(
                {"sm_group_num": 8, "manual_divisions": [[84, 24, 1]] * 6}
            )
        self.assertEqual(raw["sm_group_num"], 3)
        self.assertEqual(raw["manual_divisions"], [[80, 28, 1]])
        self.assertEqual(raw["decode_graph_mode"], "per_partition")
        self.assertTrue(raw["muxwise_allow_sm_interpolation"])

    def test_fixed_partition_override_reaches_adaptive_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pdmux.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "sm_group_num: 8",
                        "runtime_mode: improved",
                        "partition_policy: muxwise_slo",
                        "manual_divisions:",
                        "  - [92, 16, 1]",
                        "  - [86, 22, 5]",
                        "  - [78, 30, 10]",
                        "  - [66, 42, 15]",
                        "  - [52, 56, 20]",
                        "  - [46, 62, 25]",
                    )
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "SGLANG_PDMUX_CONCURRENCY_MODE": "fixed_partition",
                    "SGLANG_PDMUX_FIXED_DIVISION": "80,28",
                },
            ):
                config = load_adaptive_pdmux_config(str(config_path))

        self.assertEqual(config.sm_group_num, 3)
        self.assertEqual(config.manual_divisions, [[80, 28, 1]])
        self.assertTrue(config.muxwise_allow_sm_interpolation)

    def test_single_decode_graph_override_selects_full_sm_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pdmux.yaml"
            config_path.write_text("sm_group_num: 3\n", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {"SGLANG_PDMUX_CONCURRENCY_MODE": "single_decode_graph"},
            ):
                mode = read_decode_graph_mode(str(config_path))
                config = load_pdmux_config(str(config_path))
        self.assertEqual(mode, "full_sm_once")
        self.assertEqual(config.decode_graph_mode, "full_sm_once")

    def test_relative_artifact_paths_resolve_from_yaml_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "pdmux.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "sm_group_num: 3",
                        "runtime_mode: improved",
                        "partition_policy: bullet_srm_tpot",
                        "mi_path: ../mi",
                        "bullet_srm_profile_path: profiles/bullet.json",
                        "bullet_reference_frequency_mhz: 1234",
                    )
                ),
                encoding="utf-8",
            )
            config = load_adaptive_pdmux_config(str(config_path))

        self.assertEqual(config.mi_path, str((root / "../mi").resolve()))
        self.assertEqual(
            config.bullet_srm_profile_path,
            str((root / "profiles/bullet.json").resolve()),
        )

    def test_native_path_does_not_load_or_initialize_adaptive_runtime(self):
        scheduler = _scheduler()
        native_config = SimpleNamespace(split_forward_token_budget=65536)

        def native_init(instance):
            instance.pdmux_config = native_config

        with (
            patch(
                "sglang.srt.mi_ext.multiplex.multiplexing_dispatch_mixin.get_disagg",
                return_value=SimpleNamespace(pdmux_config_path="native.yaml"),
            ),
            patch(
                "sglang.srt.mi_ext.multiplex.multiplexing_dispatch_mixin.read_runtime_mode",
                return_value="native",
            ),
            patch(
                "sglang.srt.mi_ext.multiplex.multiplexing_dispatch_mixin.load_adaptive_pdmux_config"
            ) as load_adaptive,
            patch.object(
                SchedulerNativeMultiplexMixin,
                "init_pdmux",
                side_effect=native_init,
                autospec=True,
            ),
        ):
            scheduler.init_pdmux()

        load_adaptive.assert_not_called()
        scheduler.init_adaptive_pdmux.assert_not_called()
        self.assertIs(scheduler.pdmux_config, native_config)
        self.assertEqual((scheduler.gpu_id, scheduler.tp_size), (3, 2))

    def test_improved_path_replaces_native_config_after_stream_creation(self):
        scheduler = _scheduler()
        adaptive_config = AdaptivePDMuxConfig(runtime_mode="improved")

        def native_init(instance):
            instance.pdmux_config = SimpleNamespace()

        with (
            patch(
                "sglang.srt.mi_ext.multiplex.multiplexing_dispatch_mixin.get_disagg",
                return_value=SimpleNamespace(pdmux_config_path="improved.yaml"),
            ),
            patch(
                "sglang.srt.mi_ext.multiplex.multiplexing_dispatch_mixin.read_runtime_mode",
                return_value="improved",
            ),
            patch(
                "sglang.srt.mi_ext.multiplex.multiplexing_dispatch_mixin.load_adaptive_pdmux_config",
                return_value=adaptive_config,
            ),
            patch.object(
                SchedulerNativeMultiplexMixin,
                "init_pdmux",
                side_effect=native_init,
                autospec=True,
            ),
        ):
            scheduler.init_pdmux()

        self.assertIs(scheduler.pdmux_config, adaptive_config)
        scheduler.init_adaptive_pdmux.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
