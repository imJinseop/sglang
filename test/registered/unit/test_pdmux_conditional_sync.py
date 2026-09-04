"""CPU-only tests for PDMux conditional sync and live state observation."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.multiplex.adaptive_multiplexing_mixin import (
    SchedulerAdaptiveMultiplexMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeStream:
    def __init__(self):
        self.synchronize_calls = 0
        self.recorded_events = []
        self.waited_events = []

    def synchronize(self):
        self.synchronize_calls += 1

    def record_event(self):
        event = object()
        self.recorded_events.append(event)
        return event

    def wait_event(self, event):
        self.waited_events.append(event)


def _scheduler():
    scheduler = object.__new__(SchedulerAdaptiveMultiplexMixin)
    scheduler.sm_counts = [(108, 0), (84, 24), (78, 30), (0, 108)]
    scheduler.stream_groups = [
        (object(), object()),
        (object(), object()),
        (object(), object()),
        (object(), object()),
    ]
    scheduler.real_sm_group_num = len(scheduler.stream_groups)
    scheduler.pdmux_chunk_count = 0
    scheduler.pdmux_config = SimpleNamespace(
        partition_policy="mi",
        partition_transition="host_sync",
    )
    scheduler.model_config = SimpleNamespace(num_hidden_layers=32)
    scheduler.split_prefill_batch = None
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(update_decode_attn_backend=MagicMock())
    )
    scheduler._trace_pdmux_event = MagicMock()
    return scheduler


class TestPDMuxConditionalSync(CustomTestCase):
    def test_live_observer_reports_state_and_rate_limits_periodic_snapshots(self):
        scheduler = _scheduler()
        scheduler._pdmux_observe_interval_s = 1.0
        scheduler._pdmux_observe_last_s = float("-inf")
        scheduler._pdmux_observe_start_s = 5.0
        scheduler._pdmux_last_decision = {
            "trigger": "layer_chunk_boundary",
            "changed": False,
            "selected_stream_idx": 1,
            "decode_latency_ms": 20.0,
            "prefill_latency_ms": 28.0,
            "blended_latency_ms": 28.0,
            "bottleneck_lane": "prefill",
        }
        scheduler.split_prefill_batch = SimpleNamespace(
            is_empty=lambda: False,
            batch_size=lambda: 2,
            extend_num_tokens=4096,
            split_index=9,
        )
        running_batch = SimpleNamespace(
            is_empty=lambda: False,
            batch_size=lambda: 3,
            seq_lens_cpu=None,
            reqs=[
                SimpleNamespace(origin_input_ids=[1, 2], output_ids=[3]),
                SimpleNamespace(origin_input_ids=[1], output_ids=[2, 3]),
                SimpleNamespace(origin_input_ids=[1, 2, 3], output_ids=[]),
            ],
        )

        with (
            patch(
                "sglang.srt.multiplex.adaptive_multiplexing_mixin.time.perf_counter",
                side_effect=[10.0, 10.5],
            ),
            patch("builtins.print") as print_mock,
        ):
            scheduler._maybe_observe_pdmux_state(running_batch, stream_idx=1)
            scheduler._maybe_observe_pdmux_state(running_batch, stream_idx=1)

        print_mock.assert_called_once()
        line = print_mock.call_args.args[0]
        self.assertTrue(line.startswith("[PDMUX_STATE] "))
        payload = json.loads(line.removeprefix("[PDMUX_STATE] "))
        self.assertEqual(payload["snapshot_reason"], "periodic")
        self.assertEqual(payload["phase"], "concurrent")
        self.assertEqual(payload["decode_batch"], 3)
        self.assertEqual(payload["decode_context_tokens"], 9)
        self.assertEqual(payload["prefill_next_layer"], 9)
        self.assertEqual((payload["prefill_sm"], payload["decode_sm"]), (84, 24))
        self.assertEqual(payload["last_bottleneck_lane"], "prefill")

    def test_selection_is_side_effect_free_before_sync(self):
        scheduler = _scheduler()
        empty_batch = SimpleNamespace(is_empty=lambda: True)

        with (
            patch(
                "sglang.srt.multiplex.adaptive_multiplexing_mixin.get_current_stream_idx",
                return_value=1,
            ),
            patch(
                "sglang.srt.multiplex.adaptive_multiplexing_mixin.set_current_stream_idx"
            ) as set_stream_idx,
        ):
            selected_idx, selected_group = scheduler.adjust_stream_groups_adaptive(
                empty_batch,
                trigger="decode_batch_drained",
            )

        self.assertEqual(selected_idx, 0)
        self.assertIs(selected_group, scheduler.stream_groups[0])
        set_stream_idx.assert_not_called()
        scheduler.tp_worker.model_runner.update_decode_attn_backend.assert_not_called()

    def test_unchanged_partition_skips_both_stream_synchronizations(self):
        scheduler = _scheduler()
        prefill_stream = _FakeStream()
        decode_stream = _FakeStream()

        changed = scheduler._sync_streams_for_partition_change(
            prefill_stream,
            decode_stream,
            current_stream_idx=1,
            selected_stream_idx=1,
            trigger="layer_chunk_boundary",
        )

        self.assertFalse(changed)
        self.assertEqual(prefill_stream.synchronize_calls, 0)
        self.assertEqual(decode_stream.synchronize_calls, 0)
        scheduler._trace_pdmux_event.assert_called_once_with(
            "partition_sync_skipped",
            trigger="layer_chunk_boundary",
            stream_idx=1,
            reason="selected_partition_unchanged",
            prefill_sm=84,
            decode_sm=24,
        )

    def test_changed_partition_drains_both_old_streams(self):
        scheduler = _scheduler()
        prefill_stream = _FakeStream()
        decode_stream = _FakeStream()

        changed = scheduler._sync_streams_for_partition_change(
            prefill_stream,
            decode_stream,
            current_stream_idx=1,
            selected_stream_idx=2,
            trigger="layer_chunk_boundary",
        )

        self.assertTrue(changed)
        self.assertEqual(prefill_stream.synchronize_calls, 1)
        self.assertEqual(decode_stream.synchronize_calls, 1)
        self.assertEqual(scheduler._trace_pdmux_event.call_count, 2)
        self.assertEqual(
            scheduler._trace_pdmux_event.call_args_list[0].args,
            ("partition_sync_start",),
        )
        self.assertEqual(
            scheduler._trace_pdmux_event.call_args_list[1].args,
            ("partition_sync_end",),
        )

    def test_host_sync_always_drains_even_when_partition_is_unchanged(self):
        scheduler = _scheduler()
        scheduler.pdmux_config.partition_transition = "host_sync_always"
        prefill_stream = _FakeStream()
        decode_stream = _FakeStream()

        changed = scheduler._sync_streams_for_partition_change(
            prefill_stream,
            decode_stream,
            current_stream_idx=1,
            selected_stream_idx=1,
            trigger="new_split_prefill",
        )

        self.assertFalse(changed)
        self.assertEqual(prefill_stream.synchronize_calls, 1)
        self.assertEqual(decode_stream.synchronize_calls, 1)
        scheduler._trace_pdmux_event.assert_any_call(
            "partition_sync_start",
            trigger="new_split_prefill",
            from_stream_idx=1,
            to_stream_idx=1,
            partition_changed=False,
            prefill_sm=84,
            decode_sm=24,
        )

    def test_event_handoff_orders_new_streams_without_host_blocking(self):
        scheduler = _scheduler()
        scheduler.pdmux_config.partition_transition = "event_handoff"
        old_prefill = _FakeStream()
        old_decode = _FakeStream()
        new_prefill = _FakeStream()
        new_decode = _FakeStream()
        scheduler.stream_groups[2] = (new_prefill, new_decode)
        boundary_event = object()
        scheduler._pdmux_prefill_chunk_event = boundary_event
        scheduler._pdmux_prefill_chunk_event_stream_idx = 1

        changed = scheduler._sync_streams_for_partition_change(
            old_prefill,
            old_decode,
            current_stream_idx=1,
            selected_stream_idx=2,
            trigger="layer_chunk_boundary",
        )

        self.assertTrue(changed)
        self.assertEqual(old_prefill.synchronize_calls, 0)
        self.assertEqual(old_decode.synchronize_calls, 0)
        self.assertEqual(old_prefill.recorded_events, [])
        decode_event = old_decode.recorded_events[0]
        self.assertEqual(new_prefill.waited_events, [boundary_event, decode_event])
        self.assertEqual(new_decode.waited_events, [boundary_event, decode_event])
        scheduler._trace_pdmux_event.assert_called_once()
        self.assertEqual(
            scheduler._trace_pdmux_event.call_args.args,
            ("partition_event_handoff",),
        )

    def test_activation_updates_global_index_and_decode_backend_together(self):
        scheduler = _scheduler()

        with patch(
            "sglang.srt.multiplex.adaptive_multiplexing_mixin.set_current_stream_idx"
        ) as set_stream_idx:
            selected_group = scheduler._activate_stream_group(2)

        set_stream_idx.assert_called_once_with(2)
        scheduler.tp_worker.model_runner.update_decode_attn_backend.assert_called_once_with(
            2
        )
        self.assertIs(selected_group, scheduler.stream_groups[2])


if __name__ == "__main__":
    unittest.main()
