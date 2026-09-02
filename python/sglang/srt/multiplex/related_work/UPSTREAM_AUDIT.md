# Related-work source audit

Audited 2026-09-01.

## MuxWise

- Paper: arXiv 2504.14489v3.
- Public artifact: `ykcombat/sglang`, branch `slo_config`, commit
  `eeac5148f456428a05823c330e1d40afb0ef2438`.
- Public engine entry: `python/sglang/srt/multiplex/multiplexing.py`.
- Public partition construction: `python/sglang/srt/multiplex/pdmux_context.py`.

The artifact dispatcher only maps decode batch size through
`manual_divisions` thresholds (or `decode_bs_divisor`). Repository-wide search
found no implementation or parameters for paper Equations 1/2, no contention
grid, and no SLO-aware dispatcher. Consequently:

- `pdmux_muxwise.yaml` / mode `muxwise_artifact` preserves the public static artifact
  behavior, adapted from 132 H200 SMs to 108 A100 SMs.
- `related_work/muxwise` / mode `muxwise_model` is our inspectable paper
  reproduction and requires a measured offline profile.

## Bullet

- Paper: arXiv 2504.19516v4.
- Public runtime: `zejia-lin/BulletServe`, commit
  `445afae2abc15d578d3107d37a9b57ebc49e5e46`.
- Predictor ABI: `python/sglang/srt/bullet/shared_mng.py` loads an external
  `predictor_param_file` with `ctypes.CDLL` and calls `predic_duration` and
  `set_adaptive_num_tpcs`.
- The native provisioning call receives queue time, phase, both phase sizes,
  decode context, both TPC allocations, and remaining layers. The Python
  `ReqTimingState` separately defines estimated TTFT as predicted GPU time plus
  elapsed queue time and tracks running request TPOT from timestamps.
- Layer stepping: `python/sglang/srt/managers/tp_worker_overlap_thread.py`.
- Layer monkey patch: `python/sglang/srt/bullet/model_monkey_patch.py` and
  `python/sglang/srt/model_executor/model_runner.py`.
- Stream masks: `python/sglang/srt/bullet/sm_controller.py`.

No predictor shared library, predictor implementation, calibrated parameter
file, or profiler-to-predictor build path is in the public repository. The
runtime and scheduler shell are public; the actual SRM and the logic inside
`set_adaptive_num_tpcs` that combines request SLO state are not.
`related_work/bullet` is therefore an independent implementation of the paper
equations, not copied upstream code.

## Common-substrate limitation

The related-work policies here use one SGLang GreenContext PDMux event loop so
the policy/cost-model ablation shares model, KV pool, CUDA graphs, workload, and
energy-measurement boundaries. This is intentional, but it is not an exact
system reproduction:

- MuxWise paper-model mode uses pre-created GreenContext candidates and CUDA
  event handoff instead of the unavailable full paper scheduler.
- Bullet paper mode does not reproduce its two-process MPS + libsmctrl runtime,
  partial SM overlap, independent phase schedulers, queue reordering, or
  decode suspension. It compares the SRM-derived TPOT provisioning rule on the
  same PDMux transport.
