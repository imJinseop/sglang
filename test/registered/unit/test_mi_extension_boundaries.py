"""Regression tests for MI ownership and the extracted measurement hooks."""

import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mi_ext.llama_recording import measured_llama_layers, record_llama_step
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMIExtensionBoundaries(CustomTestCase):
    def test_custom_files_and_upstream_hooks_remain_identifiable_after_commits(self):
        root = Path(__file__).resolve().parents[3]
        manifest = json.loads(
            (root / "python/sglang/srt/mi_ext/upstream.json").read_text()
        )

        def git(*args):
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
                text=True,
            )

        try:
            git("cat-file", "-e", manifest["revision"])
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.skipTest(
                "Pinned upstream history is unavailable in this source archive"
            )
        rows = git(
            "diff", "--no-renames", "--name-status", manifest["revision"]
        ).splitlines()
        rows += [
            "A\t" + p
            for p in git("ls-files", "--others", "--exclude-standard").splitlines()
        ]
        for row in rows:
            status, path = row.split("\t", 1)
            with self.subTest(status=status, path=path):
                if path.startswith(tuple(manifest["extension_paths"])):
                    continue
                self.assertIn(path, manifest["upstream_hooks"])
                self.assertEqual(
                    status, "M", "Upstream files must not be deleted or replaced"
                )

    def test_disabled_layer_recording_preserves_original_modules(self):
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=2), layers=[object(), object()]
        )
        with patch(
            "sglang.srt.mi_ext.llama_recording.get_layerwise_ground_truth_recorder",
            return_value=None,
        ):
            self.assertIs(measured_llama_layers(model), model.layers)

    def test_layer_recording_preserves_forward_index_and_measured_inputs(self):
        layer = MagicMock(return_value=("output", "residual"))
        recorder = MagicMock()
        recorder.begin_forward.return_value = 7
        recorder.measure_layer.side_effect = lambda **kw: kw["call"]("copy_h", "copy_r")
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=1), layers=[layer]
        )
        with patch(
            "sglang.srt.mi_ext.llama_recording.get_layerwise_ground_truth_recorder",
            return_value=recorder,
        ):
            layers = measured_llama_layers(model)
            self.assertEqual(
                layers[0]("positions", "hidden", "batch", "residual"),
                ("output", "residual"),
            )
        recorder.begin_forward.assert_called_once()
        recorded = recorder.measure_layer.call_args.kwargs
        self.assertEqual((recorded["layer_index"], recorded["forward_index"]), (0, 7))
        self.assertEqual(
            (recorded["hidden_states"], recorded["residual"]), ("hidden", "residual")
        )
        layer.assert_called_once_with("positions", "copy_h", "batch", "copy_r")

    def test_step_recording_preserves_disabled_forward_and_armed_callback(self):
        forward = MagicMock(return_value="logits")
        wrapped = record_llama_step(forward)
        model = object()
        with patch.dict("os.environ", {"MI_STEP_GT_PATH": ""}):
            self.assertEqual(
                wrapped(model, "ids", "pos", "batch", "emb", True, "pp"), "logits"
            )
        forward.assert_called_once_with(model, "ids", "pos", "batch", "emb", True, "pp")
        forward.reset_mock()
        with (
            patch.dict("os.environ", {"MI_STEP_GT_PATH": "measure.json"}),
            patch(
                "e2e_inference.isolated_step_recorder.measure_step_if_armed",
                side_effect=lambda m, ids, pos, batch, call: call(
                    "copy_ids", "copy_pos"
                ),
            ),
        ):
            self.assertEqual(wrapped(model, "ids", "pos", "batch"), "logits")
            with self.assertRaisesRegex(ValueError, "token IDs"):
                wrapped(model, "ids", "pos", "batch", get_embedding=True)
        forward.assert_called_once_with(model, "copy_ids", "copy_pos", "batch")


if __name__ == "__main__":
    unittest.main()
