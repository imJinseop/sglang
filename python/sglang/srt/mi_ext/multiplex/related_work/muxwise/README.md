# MuxWise estimator and adapter

The public MuxWise artifact (`ykcombat/sglang`, branch `slo_config`) contains
the PDMux engine and a manual decode-batch-to-partition table.  It does not
contain the paper's solo-run regressors, contention-guard data, fitting code,
or SLO-aware dispatcher.

This directory keeps an independent, inspectable reproduction of the published
model separate from both native PDMux and the public-artifact threshold mode:

- `model.py`: Equations 1 and 2, the five-dimensional worst-case decode
  contention guard, JSON profile I/O, and online max-guard refinement.
- `fit_profile.py`: fits the phase regressions and contention cells from offline
  JSONL measurements.
- `policy.py`: evaluates only pre-created GreenContext candidates, reserves the
  minimum decode SM count satisfying TBT, and computes
  `ceil(T_decode * num_layers / T_prefill)` prefill layers.
- `../../adaptive_multiplexing_mixin.py`: applies that layer count and re-evaluates at
  each completed layer quantum.

No coefficients are invented.  A real Llama-3.1-8B/A100 profile must be fitted
before the paper-model policy can be used.  The existing `muxwise_artifact`
mode remains available as a clearly labelled static artifact baseline.

MuxWise does not lift a kernel cost model. Its two regressions directly predict
a complete prefill batch and a complete decode iteration at a fixed SM count.
The layer quantum is derived afterward from their latency ratio. Runtime
decode observations monotonically raise the worst-case slowdown stored in the
matching contention cell; they do not replace the solo-run regression.
