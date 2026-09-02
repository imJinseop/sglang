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
  bandwidth-contention curve, cached inverse-distance interpolation, and online
  alpha refinement.
- `fit_profile.py`: converts sparse measured execution-state JSONL rows into
  Equation 2 alpha points without any GPU dependency.
- `policy.py`: builds the current six-dimensional execution state, predicts all
  pre-created PDMux candidates, and leaves the most SMs to prefill subject to
  the decode TPOT limit.
- `../../adaptive_multiplexing_mixin.py`: executes the selected fixed number of prefill
  layers, asks the policy again at the next layer boundary, and feeds completed
  CUDA-event measurements back to the SRM.

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
