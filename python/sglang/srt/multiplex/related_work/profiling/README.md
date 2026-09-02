# Common related-work profiler

This directory measures MuxWise and Bullet on one PDMux transport so model,
KV cache, CUDA graphs, candidate partitions, scenario, and energy boundaries
stay identical.

- `collector.py` records exact batch vectors, layer range, partition, phase,
  concurrency state, and CUDA-event latency as JSONL.
- `generate_scenario.py` creates deterministic solo/concurrent fitting traffic,
  including grouped prompts that exercise Radix prefix reuse.
- `build_profiles.py` converts one common observation set into the MuxWise
  regressions/contention guard and Bullet SRM alpha profile.
- `analyze_validation.py` reports held-out GPU-quantum prediction error,
  partition choices, decision overhead, and end-to-end request metrics.

Keep profiling, validation, and final evaluation scenarios disjoint.  Repeat
the fixed execution-state matrix rather than treating thousands of correlated
decode iterations from one serving replay as independent samples.  All methods
must use the same fixed GreenContext menu and SLO during end-to-end comparison.

The Bullet phase predictor estimates GPU service quanta.  It does not by itself
predict request TTFT or TPOT: queue delay, request progress, inter-iteration
gaps, and P90 request state must be added by a full scheduler-level model.
