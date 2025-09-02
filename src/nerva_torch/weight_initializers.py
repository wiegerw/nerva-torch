# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Weight and bias initialization helpers for linear layers."""

import torch
from nerva_torch.matrix_operations import Matrix


def set_bias_to_zero(b: Matrix):
    """Set all bias values to zero."""
    b.data.zero_()


def set_weights_xavier(W: Matrix):
    """Initialize weights using Xavier/Glorot initialization."""
    K, D = W.shape
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (K + D)))
    W.data = torch.randn(K, D) * xavier_stddev


def set_bias_xavier(b: Matrix):
    """Set bias to zero (Xavier scheme for bias)."""
    set_bias_to_zero(b)


def set_weights_xavier_normalized(W: Matrix):
    """Initialize weights using normalized Xavier initialization."""
    K, D = W.shape
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (K + D)))
    random_matrix = torch.randn(K, D)
    W.data = random_matrix * xavier_stddev


def set_bias_xavier_normalized(b: Matrix):
    """Set bias to zero (normalized Xavier scheme)."""
    set_bias_to_zero(b)


def set_weights_he(W: Matrix):
    """Initialize weights using He initialization for ReLU networks."""
    K, D = W.shape
    he_stddev = torch.sqrt(torch.tensor(2.0 / D))
    random_matrix = torch.randn(K, D)
    W.data = random_matrix * he_stddev


def set_bias_he(b: Matrix):
    """Set bias to zero (He scheme for bias)."""
    set_bias_to_zero(b)


def set_layer_weights(layer, text: str):
    """Initialize a layer's parameters according to a named scheme."""
    if text == 'Xavier':
        set_weights_xavier(layer.W)
        set_bias_xavier(layer.b)
    elif text == 'XavierNormalized':
        set_weights_xavier_normalized(layer.W)
        set_bias_xavier_normalized(layer.b)
    elif text == 'He':
        set_weights_he(layer.W)
        set_bias_he(layer.b)
    else:
        raise RuntimeError(f'Could not parse weight initializer "{text}"')
