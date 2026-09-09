# MI-owned SGLang extensions

All implementation in this package belongs to the MI project. See
[`MI_EXTENSIONS.md`](../../../../MI_EXTENSIONS.md) for the pinned upstream
revision, integration hooks, module map and commit order.

The package initializer is intentionally lightweight: importing the namespace
does not initialize CUDA, load a model, or install global monkey patches.
