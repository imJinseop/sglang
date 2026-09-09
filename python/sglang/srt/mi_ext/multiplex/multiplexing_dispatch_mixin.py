# MI project extension; not part of upstream SGLang.
"""Thin opt-in dispatcher between native and adaptive PDMux runtimes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mi_ext.multiplex.adaptive_multiplexing_mixin import (
    SchedulerAdaptiveMultiplexMixin,
)
from sglang.srt.mi_ext.multiplex.adaptive_pdmux_config import (
    load_adaptive_pdmux_config,
    read_runtime_mode,
)
from sglang.srt.multiplex.multiplexing_mixin import (
    SchedulerMultiplexMixin as SchedulerNativeMultiplexMixin,
)
from sglang.srt.runtime_context import get_disagg

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SchedulerMultiplexDispatchMixin(
    SchedulerNativeMultiplexMixin,
    SchedulerAdaptiveMultiplexMixin,
):
    """Preserve native behavior unless a config explicitly opts in."""

    def init_pdmux(self: Scheduler):
        config_path = get_disagg().pdmux_config_path
        runtime_mode = read_runtime_mode(config_path)
        adaptive_config = (
            load_adaptive_pdmux_config(config_path)
            if runtime_mode == "improved"
            else None
        )

        # The pinned upstream PDMux mixin still uses these legacy names while
        # the current Scheduler stores them under ParallelState.  Aliasing here
        # keeps the native source and its control flow untouched.
        self.gpu_id = self.ps.gpu_id
        self.tp_size = self.ps.tp_size
        SchedulerNativeMultiplexMixin.init_pdmux(self)

        if adaptive_config is not None:
            self.pdmux_config = adaptive_config
            self.init_adaptive_pdmux()

    def event_loop_pdmux(self: Scheduler):
        if getattr(self.pdmux_config, "runtime_mode", "native") == "improved":
            return self.event_loop_pdmux_adaptive()
        return SchedulerNativeMultiplexMixin.event_loop_pdmux(self)
