# MI project extensions

This checkout contains MI-owned additions to SGLang. They remain identifiable
after committing: custom implementation lives under
[`python/sglang/srt/mi_ext/`](python/sglang/srt/mi_ext/), and custom tests use the
`test_mi_` filename prefix. These are independently implemented adapters and
experiments, not upstream SGLang or the original MuxWise/Bullet code.

The last upstream revision before the MI commits is
`9d18277c2720f3a1d6e64f90259e1d1b9ba9a26a`. Its identity and the allowed upstream
hook files are recorded in
[`mi_ext/upstream.json`](python/sglang/srt/mi_ext/upstream.json).
Keep this baseline when making ordinary custom commits. Update it explicitly
when adopting a new upstream revision.

From this submodule directory, inspect the full custom delta at any time:

```bash
git diff --stat 9d18277c2720f3a1d6e64f90259e1d1b9ba9a26a
git diff 9d18277c2720f3a1d6e64f90259e1d1b9ba9a26a -- python/sglang/srt/managers python/sglang/srt/model_executor python/sglang/srt/models python/sglang/srt/multiplex
git log --oneline 9d18277c2720f3a1d6e64f90259e1d1b9ba9a26a..HEAD
```

## Implementation map

| MI-owned path | Responsibility |
| --- | --- |
| `mi_ext/multiplex/` | Native/adaptive dispatch, runtime loop, policy inputs, Ours adapter |
| `mi_ext/multiplex/related_work/bullet/` | Independent Bullet SRM, calibration and scheduling |
| `mi_ext/multiplex/related_work/muxwise/` | Independent MuxWise model, calibration and scheduling |
| `mi_ext/multiplex/related_work/profiling/` | Shared calibration collection and fitting tools |
| `mi_ext/multiplex/context.py` | Concurrency configuration and OmniDecode stream setup |
| `mi_ext/decode_graph.py` | Decode graph routing and capture statistics |
| `mi_ext/llama_recording.py` | Optional Llama layer/step measurement hooks |
| `mi_ext/layerwise_ground_truth.py` | Ground-truth recording and measurement dispatch |
| `test/registered/unit/test_mi_*.py` | MI extension regression and ownership tests |

Five existing upstream files retain small integration hooks, listed in the
manifest. Implement behavior in `mi_ext` and keep those call sites small.
`tp_worker.py` matches the pinned upstream version. The extension boundary test
checks new files and upstream edits against this manifest, including changes
that have already been committed.

The parent MI checkout provides cost models, device profiles and measurement
entrypoints. Their active imports use `sglang.srt.mi_ext`. The former custom
module locations under `srt/multiplex` and `srt/utils` have been removed; update
external scripts to the new namespace. Recorded experiment JSON retains the
historical paths and source hashes from the actual run.

## Commit order

Commit and push this submodule's `custom_pdmux_v0.5.18` branch first. Then stage
`3rdparty/sglang` in the parent MI repository and commit the parent changes. The
parent stores the exact submodule commit SHA; the branch entry in `.gitmodules`
does not replace that pinned SHA. Push the parent after the referenced submodule
commit is available from the configured fork.
