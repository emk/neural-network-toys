# This file is used to generate test data for our Rust code. This is useful
# because:
#
# 1. It's easy to make small errors when calculating derivatives or
#    implementing the layers.
# 2. Many small errors will pass unnoticed, resulting in either worse
#    performance or oscillating/exploding gradients.
#
# To use this file, run:
#
#    pipenv install
#    pipenv run python make_gradient_test_data.py`.

import json
import os
import torch
from torch import Tensor
import torch.nn.functional as F

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")


def mse(outputs: Tensor, targets: Tensor) -> Tensor:
    # The Rust code computes a vector of MSEs for each input/target pair, not
    # the MSE across all the pairs at once. So in order to get the same
    # gradients, we need to compute the same MSEs and sum them.
    mean_dims = list(range(1, len(outputs.shape)))
    return ((outputs - targets) ** 2).mean(dim=mean_dims).sum()


def catagorical_cross_entropy(outputs: Tensor, targets: Tensor) -> Tensor:
    # Compute element-wise cross entropy values.
    temp = -1 * targets * outputs.log()

    # Sum across all the classes, to mimic Rust. Then sum across all the
    # examples, because PyTorch can only compute the gradient of a scalar.
    return temp.sum(dim=1).sum()


def fully_connected_data() -> dict:
    def fully_connected(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return x @ w + b

    num_features = 3
    num_outputs = 2
    num_examples = 4

    x = torch.randn(num_examples, num_features, device=device, requires_grad=True)
    w = torch.randn(num_features, num_outputs, device=device, requires_grad=True)
    b = torch.randn(num_outputs, device=device, requires_grad=True)

    y = fully_connected(x, w, b)

    targets = torch.zeros(num_examples, num_outputs, device=device)
    loss = mse(y, targets)
    loss.backward()

    return {
        "inputs": x.tolist(),
        "weights": w.tolist(),
        "bias": b.tolist(),
        "outputs": y.tolist(),
        "targets": targets.tolist(),
        "gradients": {
            "inputs": x.grad.tolist(),
            "weights": w.grad.tolist(),
            "bias": b.grad.tolist(),
        },
    }


def conv2d_data() -> dict:
    def conv2d(inputs: Tensor, filters: Tensor, biases: Tensor) -> Tensor:
        return F.conv2d(inputs, filters, biases, padding="same")

    img_size = 4
    filter_size = 3
    num_channels = 1
    num_filters = 2
    num_examples = 2

    inputs = torch.randn(
        num_examples,
        num_channels,
        img_size,
        img_size,
        device=device,
        requires_grad=True,
    )
    filters = torch.randn(
        num_filters,
        num_channels,
        filter_size,
        filter_size,
        device=device,
        requires_grad=True,
    )
    biases = torch.randn(num_filters, device=device, requires_grad=True)

    outputs = conv2d(inputs, filters, biases)
    outputs.retain_grad()

    targets = torch.zeros(num_examples, num_filters, img_size, img_size, device=device)
    loss = mse(outputs, targets)
    loss.backward()

    return {
        "inputs": inputs.tolist(),
        "filters": filters.tolist(),
        "biases": biases.tolist(),
        "outputs": outputs.tolist(),
        "targets": targets.tolist(),
        "gradients": {
            "outputs": outputs.grad.tolist(),
            "inputs": inputs.grad.tolist(),
            "filters": filters.grad.tolist(),
            "biases": biases.grad.tolist(),
        },
    }


def pool2d_data() -> dict:
    def pool2d(inputs: Tensor, pool_size: int) -> Tensor:
        return F.max_pool2d(inputs, pool_size, stride=pool_size)

    img_size = 4
    pool_size = 2
    num_channels = 1
    num_examples = 2

    inputs = torch.randn(
        num_examples,
        num_channels,
        img_size,
        img_size,
        device=device,
        requires_grad=True,
    )

    outputs = pool2d(inputs, pool_size)

    targets = torch.zeros(
        num_examples, num_channels, img_size // 2, img_size // 2, device=device
    )
    loss = mse(outputs, targets)
    loss.backward()

    return {
        "inputs": inputs.tolist(),
        "outputs": outputs.tolist(),
        "targets": targets.tolist(),
        "kernel_size": pool_size,
        "stride": pool_size,
        "gradients": {
            "inputs": inputs.grad.tolist(),
        },
    }


def leaky_relu_data() -> dict:
    def leaky_relu(inputs: Tensor) -> Tensor:
        return F.leaky_relu(inputs, negative_slope=0.01)

    num_features = 3
    num_examples = 3

    x = torch.randn(num_examples, num_features, device=device, requires_grad=True)

    y = leaky_relu(x)

    targets = torch.zeros(num_examples, num_features, device=device)
    loss = mse(y, targets)
    loss.backward()

    return {
        "inputs": x.tolist(),
        "outputs": y.tolist(),
        "targets": targets.tolist(),
        "gradients": {
            "inputs": x.grad.tolist(),
        },
    }


def tanh_data() -> dict:
    def tanh(inputs: Tensor) -> Tensor:
        return torch.tanh(inputs)

    num_features = 3
    num_examples = 2

    x = torch.randn(num_examples, num_features, device=device, requires_grad=True)

    y = tanh(x)

    targets = torch.zeros(num_examples, num_features, device=device)
    loss = mse(y, targets)
    loss.backward()

    return {
        "inputs": x.tolist(),
        "outputs": y.tolist(),
        "targets": targets.tolist(),
        "gradients": {
            "inputs": x.grad.tolist(),
        },
    }


def softmax_data() -> dict:
    def softmax(inputs: Tensor) -> Tensor:
        return F.softmax(inputs, dim=1)

    num_features = 3
    num_examples = num_features  # Needed for eye() to work.

    x = torch.randn(num_examples, num_features, device=device, requires_grad=True)

    y = softmax(x)

    targets = torch.eye(num_examples, num_features, device=device)
    loss = catagorical_cross_entropy(y, targets)
    loss.backward()

    return {
        "inputs": x.tolist(),
        "outputs": y.tolist(),
        "targets": targets.tolist(),
        "gradients": {
            "inputs": x.grad.tolist(),
        },
    }


# Our inputs, outputs, and gradients as JSON
json_data = {
    "fully_connected": fully_connected_data(),
    "conv2d": conv2d_data(),
    "pool2d": pool2d_data(),
    "leaky_relu": leaky_relu_data(),
    "tanh": tanh_data(),
    "softmax": softmax_data(),
}

# Write each layer's JSON data to a separate file.
os.makedirs("../fixtures/layers", exist_ok=True)
for layer, data in json_data.items():
    with open(f"../fixtures/layers/{layer}.json", "w") as f:
        json.dump(data, f, indent=2)
