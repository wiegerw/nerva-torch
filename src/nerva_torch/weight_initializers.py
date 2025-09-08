# Copyright 2023 - 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Weight and bias initialization helpers for linear layers."""

import math
from nerva_torch.utilities import parse_function_call
from nerva_torch.matrix_operations import Matrix


def set_bias_zero(b: Matrix):
    """Set all bias values to zero."""
    b.data.zero_()


def set_bias_uniform(b_: Matrix, a: float = 0.0, b: float = 1.0):
    """Uniform initialization within [a, b)."""
    b_.uniform_(a, b)


def set_bias_normal(b: Matrix, mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) initialization with given mean and std."""
    b.normal_(mean, std)


def set_weights_uniform(W: Matrix, a: float = 0.0, b: float = 1.0):
    """Uniform initialization within [a, b)."""
    W.uniform_(a, b)


def set_weights_normal(W: Matrix, mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) initialization with given mean and std."""
    W.normal_(mean, std)


def set_weights_xavier_uniform(W: Matrix):
    """Xavier / Glorot uniform initialization (for tanh/sigmoid).

    K = fan-out (output size)
    D = fan-in  (input size)
    """
    K, D = W.shape
    limit = math.sqrt(6.0 / (D + K))  # sqrt(6 / (fan_in + fan_out))
    W.data.uniform_(-limit, limit)


def set_weights_xavier_normal(W: Matrix):
    """Xavier / Glorot normal initialization (for tanh/sigmoid).

    K = fan-out (output size)
    D = fan-in  (input size)
    """
    K, D = W.shape
    std = math.sqrt(2.0 / (D + K))  # sqrt(2 / (fan_in + fan_out))
    W.normal_(0.0, std)


def set_weights_he_normal(W: Matrix):
    """He / Kaiming normal initialization (for ReLU).

    K = fan-out (output size)
    D = fan-in  (input size)
    """
    K, D = W.shape
    std = math.sqrt(2.0 / D)  # sqrt(2 / fan_in)
    W.data.normal_(0.0, std)


def set_weights_he_uniform(W: Matrix):
    """He / Kaiming uniform initialization (for ReLU, less common).

    K = fan-out (output size)
    D = fan-in  (input size)
    """
    K, D = W.shape
    limit = math.sqrt(6.0 / D)  # sqrt(6 / fan_in)
    W.data.uniform_(-limit, limit)


def set_weights_zero(W: Matrix):
    """Initialize weights to zero.

    Note: Initializing all weights to zero is generally not recommended because
    it causes all neurons to learn the same features during training, leading to
    symmetry that prevents effective learning and updates (the "symmetry breaking" problem).
    This initializer can be useful for biases or special cases but should be avoided for weights.
    """
    W.zero_()


def set_layer_weights(layer, text: str):
    """Initialize a layer's parameters according to a named scheme."""
    func = parse_function_call(text)
    if func.name == 'Uniform':
        a = func.as_scalar('a', 0)
        b = func.as_scalar('b', 1)
        set_weights_uniform(layer.W, a, b)
        set_bias_zero(layer.b)
    elif func.name == 'Normal':
        a = func.as_scalar('a', 0)
        b = func.as_scalar('b', 1)
        set_weights_normal(layer.W, a, b)
        set_bias_zero(layer.b)
    if func.name == 'XavierUniform':
        set_weights_xavier_uniform(layer.W)
        set_bias_zero(layer.b)
    elif func.name == 'XavierNormal':
        set_weights_xavier_normal(layer.W)
        set_bias_zero(layer.b)
    elif func.name == 'HeUniform':
        set_weights_he_uniform(layer.W)
        set_bias_zero(layer.b)
    elif func.name == 'HeNormal':
        set_weights_he_normal(layer.W)
        set_bias_zero(layer.b)
    elif func.name == 'Zero':
        set_weights_zero(layer.W)
        set_bias_zero(layer.b)
    else:
        raise RuntimeError(f'Could not parse weight initializer "{text}"')
