# MI project extension; not part of upstream SGLang.
"""Optional layer/step measurements exposed through small Llama hooks."""

import os
from functools import wraps

from .layerwise_ground_truth import get_layerwise_ground_truth_recorder


class _MeasuredLayers:
    def __init__(self, layers, recorder):
        self.layers = layers
        self.recorder = recorder
        self.forward_index = recorder.begin_forward()

    def __getitem__(self, index):
        layer = self.layers[index]

        def run(positions, hidden_states, forward_batch, residual):
            return self.recorder.measure_layer(
                layer_index=index,
                forward_index=self.forward_index,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                call=lambda hidden, res: layer(positions, hidden, forward_batch, res),
            )

        return run


def measured_llama_layers(model):
    recorder = get_layerwise_ground_truth_recorder(model.config.num_hidden_layers)
    if recorder is None:
        return model.layers
    return _MeasuredLayers(model.layers, recorder)


def record_llama_step(forward):
    @wraps(forward)
    def run(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        if not os.environ.get("MI_STEP_GT_PATH"):
            return forward(
                self,
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                get_embedding,
                pp_proxy_tensors,
            )
        if input_embeds is not None or get_embedding or pp_proxy_tensors is not None:
            raise ValueError(
                "step GT requires token IDs and next-token logits on one GPU"
            )
        from e2e_inference.isolated_step_recorder import measure_step_if_armed

        return measure_step_if_armed(
            self,
            input_ids,
            positions,
            forward_batch,
            lambda ids, pos: forward(self, ids, pos, forward_batch),
        )

    return run
