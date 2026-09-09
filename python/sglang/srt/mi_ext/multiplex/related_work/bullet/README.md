# Bullet adapter boundary

This directory owns only the Bullet scheduling adapter that runs inside the
SGLang PDMux event loop.  It is deliberately separate from native PDMux.

The public BulletServe repository exposes its runtime predictor through the
native ABI functions `predic_duration` and `set_adaptive_num_tpcs`, but does not
ship the predictor implementation or its calibrated parameter library. The
performance model used here is therefore an independent paper reproduction:

- `types.py`: CUDA-free execution state, fixed plan, kernel, and lane types.
- `workloads.py`: converts MI's Llama kernel-work description into the
  per-kernel work of one Transformer layer. MI does not predict latency here.
- `srm.py`: SM-scaling kernel roofline, sparse alpha calibration, decode
  bandwidth-contention curve, local linear alpha interpolation/extrapolation, and online
  alpha refinement.
- `fit_profile.py`: converts sparse measured execution-state JSONL rows into
  Equation 2 alpha points without any GPU dependency.
- `policy.py`: builds the current six-dimensional execution state, predicts all
  pre-created PDMux candidates, and leaves the most SMs to prefill subject to
  the decode TPOT limit.
- `../../adaptive_multiplexing_mixin.py`: executes the selected fixed number of prefill
  layers, synchronizes the prefill stream, and requests a new policy decision
  before the next chunk. This happens even with `repartition_interval_chunks=0`.
  Completed CUDA-event measurements are offered to the SRM; online updates
  remain controlled separately by `bullet_online_calibration`.

## Prefill scheduling cadence

[Section 3.3.1](https://arxiv.org/html/2504.19516v4#S3.SS3.SSS1) specifies a
fixed number of prefill layers per CPU scheduling step. The installed device
profile calibrates four layers, so both paired and prefill-only execution use
`min(4, remaining_layers)`. The token-budget heuristic is not used for Bullet.

After each chunk is submitted, the prefill stream is synchronized before
requesting reevaluation. The existing shared PDMux loop then consumes the
current decode result and, on its next iteration, builds fresh batch/context
state and evaluates all candidate partitions. If the selected pair is unchanged,
the chunk boundary adds no partition barrier. If it changes, both old lanes are
host-synchronized before activating the new Green Context pair and decode
backend. These Bullet chunk boundaries use that rule regardless of the generic
partition-handoff mode. The final chunk uses the existing prefill-result/merge
path and its subsequent partition decision.

This restores chunk-by-chunk control within the shared loop; it does not add an
independent prefill CPU scheduler that advances while the loop waits for decode.
The paper's separate engines and overlapping-SM reconfiguration are different
from this requested two-lane Green Context barrier. Historical runs with
interval zero before this change did not perform chunk-boundary reevaluation.

## From kernel to layer and iteration

For each kernel `k`, `srm.py` computes
`max(FLOPs_k / C_p, bytes_k / D_p)` and sums those values. `workloads.py`
normalizes the full Llama work to one Transformer layer. `policy.py` multiplies
that unit by the fixed prefill-layer step, while decode is multiplied by all 32
Llama-3.1-8B layers. The result is one prefill quantum and one complete decode
iteration, which is exactly the granularity needed by the scheduler.

With no `bullet_srm_profile_path`, alpha and contention default to 1.0. That is
useful only as an uncalibrated smoke test. Do not report its benchmark numbers
as Bullet results; first collect sparse per-execution-state latency and decode
bandwidth-contention measurements.

The predicted decode quantum is GPU execution time for one full-model decode
iteration, not end-to-end request TPOT. Request TPOT additionally contains
inter-iteration scheduling/launch gaps, and TTFT additionally contains queue
time and all remaining prefill steps. The complete paper scheduler combines
SRM with per-request arrival/decode-start timestamps, prefill queue/in-flight
sets, executed-layer progress, and P90 SLO checks. This common-PDMux adapter
does not yet reproduce that request-level control plane; its TPOT comparison is
therefore diagnostic until those states are integrated.

This is not the authors' unreleased SRM binary and it does not reproduce
Bullet's two-process MPS/libsmctrl engine.  The transport remains SGLang
GreenContext PDMux so cost models and dispatch policies can be compared on one
common execution substrate.

Device-local calibration is available at
`python -m e2e_inference.related_work_calibration run --target bullet` in the
parent MI checkout. It measures capacities and SM-specific UG+memory-copy
contention without MI latency coefficients, then calibrates actual SGLang
control quanta and validates disjoint workload states. Installed profiles
declare their measured prefill-layer quantum; the policy rejects a mismatch.
