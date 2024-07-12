# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)


import torch


def mean_squared_error_loss_torch(Y, T):
    """
    Computes the mean squared error loss between Y and T.

    Parameters:
    Y (torch.Tensor): The predicted values.
    T (torch.Tensor): The target values.

    Returns:
    float: The computed loss.
    """
    if Y.shape != T.shape:
        raise ValueError("The shapes of Y and T must be the same")

    loss = torch.nn.MSELoss(reduction='sum')
    return loss(Y, T).item() / (Y.numel())


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
    return loss(Y, T)


def negative_log_likelihood_loss_torch(Y, T):
    """
    Computes the negative log likelihood loss between Y and T.

    Parameters:
    Y (torch.Tensor): The predicted values.
    T (torch.Tensor): The target values.

    Returns:
    float: The computed loss.
    """

    YT = torch.sum(Y * T, dim=1)
    loss = -torch.sum(torch.log(YT))

    return loss.item()


def cross_entropy_loss_torch(Y, T):
    """
    Computes the cross entropy loss between Y and T.

    Parameters:
    Y (torch.Tensor): The predicted probabilities.
    T (torch.Tensor): The one-hot encoded target probabilities.

    Returns:
    float: The computed loss.
    """

    if Y.shape != T.shape:
        raise ValueError("The shapes of Y and T must be the same")

    loss = -torch.sum(T * torch.log(Y))
    return loss


def logistic_cross_entropy_loss_torch(Y, T):
    """
    Computes the logistic cross entropy loss between Y and T.

    Parameters:
    Y (torch.Tensor): The predicted values.
    T (torch.Tensor): The target values.

    Returns:
    float: The computed loss.
    """

    if Y.shape != T.shape:
        raise ValueError("The shapes of Y and T must be the same")

    sigmoid_Y = torch.sigmoid(Y)
    loss = -torch.dot(T.view(-1), torch.log(sigmoid_Y.view(-1)))

    return loss.item()