# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Thin wrappers around PyTorch loss modules for comparison and testing."""

import torch
from nerva_torch.datasets import from_one_hot


def squared_error_loss_torch(Y, T):
    """
    Computes the squared error loss between Y and T.

    Parameters:
    Y (torch.Tensor): The predicted values.
    T (torch.Tensor): The target values.

    Returns:
    float: The computed loss.
    """
    if Y.shape != T.shape:
        raise ValueError("The shapes of Y and T must be the same")

    loss = torch.nn.MSELoss(reduction='sum')
    return loss(Y, T).item()


def softmax_cross_entropy_loss_torch(Y, T):
    """
    Computes the softmax cross entropy loss between Y and T.

    Parameters:
    Y (torch.Tensor): The predicted values.
    T (torch.Tensor): The target values.

    Returns:
    float: The computed loss.
    """

    if Y.shape != T.shape:
        raise ValueError("The shapes of Y and T must be the same")

    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    return loss(Y, T).item()


def negative_likelihood_loss_torch(Y, T):
    """
    Computes the negative likelihood loss between Y and T. Note that PyTorch does
    not apply the log function, since it assumes Y is the output of a log softmax
    layer. For this reason we omit "log" in the name.

    Parameters:
    Y (torch.Tensor): The predicted values.
    T (torch.Tensor): The target values.

    Returns:
    float: The computed loss.
    """

    loss = torch.nn.NLLLoss(reduction='sum')
    return loss(Y, from_one_hot(T)).item()
