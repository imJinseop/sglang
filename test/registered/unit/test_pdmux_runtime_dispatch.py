"""CPU-only contracts for keeping native and adaptive PDMux isolated."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.multiplex.adaptive_pdmux_config import (
    AdaptivePDMuxConfig,
    load_adaptive_pdmux_config,
)
from sglang.srt.multiplex.multiplexing_dispatch_mixin import (
    SchedulerMultiplexDispatchMixin,
    SchedulerNativeMultiplexMixin,
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
                "sglang.srt.multiplex.multiplexing_dispatch_mixin.get_disagg",
                return_value=SimpleNamespace(pdmux_config_path="native.yaml"),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_dispatch_mixin.read_runtime_mode",
                return_value="native",
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_dispatch_mixin.load_adaptive_pdmux_config"
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
                "sglang.srt.multiplex.multiplexing_dispatch_mixin.get_disagg",
                return_value=SimpleNamespace(pdmux_config_path="improved.yaml"),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_dispatch_mixin.read_runtime_mode",
                return_value="improved",
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_dispatch_mixin.load_adaptive_pdmux_config",
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
