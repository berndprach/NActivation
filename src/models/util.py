from typing import Optional

import torch
from torch import nn


def get_all_layers(model: nn.Sequential):
    all_layers = []
    for layer in model:
        if isinstance(layer, nn.Sequential):
            all_layers.extend(get_all_layers(layer))
        else:
            all_layers.append(layer)
    return all_layers


def get_operator_norms_of_input_gradients(model,
                                          input_batch,
                                          nrof_inputs: Optional[int] = None,
                                          ):
    operator_norms = []
    input_batch.requires_grad = True
    if nrof_inputs is None:
        nrof_inputs = input_batch.shape[0]
    else:
        nrof_inputs = min(nrof_inputs, input_batch.shape[0])

    for i in range(nrof_inputs):
        single_input = input_batch[i:i + 1]
        jacobian = torch.autograd.functional.jacobian(model, single_input)
        # jacobian.shape should be (1, output_dim[1:], 1, input_dim[1:])
        assert len(jacobian.shape) == len(single_input.shape) + 2
        # Assuming model output is a batch of vectors!
        jacobian = jacobian[0, :, 0]
        jacobian_2d = jacobian.reshape(jacobian.shape[0], -1)
        input_gradient_norm = torch.linalg.matrix_norm(jacobian_2d, ord=2)
        operator_norms.append(input_gradient_norm)
    return torch.stack(operator_norms)


def get_summary(model):
    nrof_trainable_paras = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return (
        f"Number of trainable parameters: {nrof_trainable_paras:,}\n\n"
        f"{model}\n"
    )


def get_activation_iterator(model, input_batch, include_input=False):
    current_activation = input_batch
    with torch.no_grad():
        if include_input:
            yield "Input", current_activation
        for layer in get_all_layers(model):
            current_activation = layer(current_activation)
            layer_name = layer.__class__.__name__
            yield layer_name, current_activation


def plot_activation_variance(activation_variances):
    import matplotlib.pyplot as plt

    log2_variances = torch.log2(torch.tensor(activation_variances))
    min_log2_variance = log2_variances.min()
    max_log2_variance = log2_variances.max()
    y_ticks = torch.arange(int(min_log2_variance),
                           int(max_log2_variance) + 2)

    plt.figure(figsize=(10, 5))
    plt.title("Activation variance")
    plt.plot(log2_variances)
    plt.scatter(torch.arange(len(log2_variances)), log2_variances)

    plt.ylabel("Activation variance (Log2 scale)")
    plt.yticks(y_ticks, [f"$2^{{{y}}}$" for y in y_ticks])
    plt.grid(True, which="major", axis="both")
    plt.tight_layout()


def activation_variance_to_string(layer_names, activation_variances):
    outlines = []
    previous_variance = None
    for i, (ln, var) in enumerate(zip(layer_names, activation_variances)):
        if previous_variance is None:
            previous_variance = var
        proportion = var / (previous_variance + 1e-8)
        outlines.append(f"{i: 3d}, {ln:<30.30}: {var:6.3g} ({proportion:.1%})")
        previous_variance = var
    return "\n".join(outlines)


def get_activation_variance(model, input_batch):
    variances = []
    layer_names = []
    for layer_name, activation in get_activation_iterator(model, input_batch):
        layer_names.append(layer_name)
        variances.append(activation.var(dim=0, correction=0).sum().item())
    return layer_names, variances
