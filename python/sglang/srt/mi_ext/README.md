# MI-owned SGLang extensions

All implementation in this package belongs to the MI project. See
[`MI_EXTENSIONS.md`](../../../../MI_EXTENSIONS.md) for the pinned upstream
revision, integration hooks, module map and commit order.

The package initializer is intentionally lightweight: importing the namespace
does not initialize CUDA, load a model, or install global monkey patches.

`decode_graph.py` captures only stream groups with nonzero decode SMs, retaining
their original indices for graph replay. An eight-group menu (prefill-only, six
split pairs, decode-only) therefore has seven capture streams. Eight batch
buckets produce 56 decode graphs in partitioned mode; full-SM-once mode produces
eight. The unused dummy decode stream in the prefill-only endpoint is excluded.
Historical results with 64 graphs retain the actual count measured by that code.

## Ours serving policy

`multiplex/mi_pdmux_policy.py` connects the parent repository's
`bullet_green.mi_sglang_runtime.MISGLangRuntimePredictor` to the existing
candidate stream menu. `LLMWorkModel` compiles the shared SGLang plan once into
numeric kernel descriptors and layer-range multiplicities. Runtime fills reusable
operation/access buffers for a complete decode step and the current prefill layer
chunk. `PreparedCostModel` binds global coefficients to those counts once per
state/SM, then clock search evaluates only the prepared numerical equations.
Input embedding and next-token logits costs follow the actual layer boundaries.
CPU compilation, device curves and scratch allocation happen at initialization.
Runtime does not create Workload/Simulator/report objects or expand model layers.
The full-result cache is optional reuse of identical inputs; novel-input speed
does not depend on it. This adds no fit coefficients or online model updates.

`--policy mi --tpot-slo-ms 30` selects Ours through the parent's serving runner.
It selects the smallest decode SM allocation meeting predicted TPOT at a common
power-feasible clock; the remaining SMs go to prefill. The runner reads the actual
NVML power cap. Each candidate recomputes costs across supported clocks at or below
the requested frequency. If no candidate meets TPOT, it selects the lowest decode
latency among power-feasible candidates and records the miss. No EDP objective or
objective flag remains. Original reselection triggers and token budget 65536 are
the defaults. `--reevaluate-every-prefill-layer-chunks 1` adds reevaluation after
each completed prefill layer chunk; `0` disables this extra trigger. Decode keeps
running while the completion event is pending. At this extra boundary, the same
pair is retained; a changed pair drains both lanes before activation. Native
trigger host handoff is unchanged. No CUDA context is created at selection.

The model scope is GPU model forward to FP32 next-token logits. It excludes
sampling, scheduler/CPU launch time and KV metadata preparation. Sequence
lengths currently use the adapter's homogeneous projection (ceiling of the
batch mean); heterogeneous attention is approximate. Prefix-reused prefill and
unsupported attention dispatches raise explicitly, retaining the shared model's
coverage contract. The two-lane adapter combines solo predictions with a
shared-bandwidth and mean busy-lane power approximation. It adds the dynamic
energy/time rate of each lane and counts static power once. This permits repeated
decode execution during one long prefill chunk. Clock bisection assumes locally
monotone average power demand; the selected clock is always evaluated and checked.
It does not model kernel-scale governor transients. The adapter
and treats SM oversubscription's complementary SM share as an estimate of interference,
not a hardware-enforced decode partition. Model feasibility is not a measured
SLO guarantee. The serving runner automatically writes sparse predictions,
candidate costs, decision CPU time, and a report comparing predicted clocks with
NVML samples during active concurrent intervals. `--trace-pdmux-events` additionally
records detailed execution/synchronization events for inspection.
