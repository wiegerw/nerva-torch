# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch

from nerva_torch.utilities import parse_function_call

Matrix = torch.Tensor


def Relu(X: Matrix):
    return torch.max(torch.zeros_like(X), X)


def Relu_gradient(X: Matrix):
    return torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))


def Leaky_relu(alpha):
    return lambda X: torch.max(X, alpha * X)


def Leaky_relu_gradient(alpha):
    return lambda X: torch.where(X > 0, torch.ones_like(X), torch.full_like(X, alpha))


def All_relu(alpha):
    return lambda X: torch.where(X < 0, alpha * X, X)


def All_relu_gradient(alpha):
    return lambda X: torch.where(X < 0, alpha, 1)


def Hyperbolic_tangent(X: Matrix):
    return torch.tanh(X)


def Hyperbolic_tangent_gradient(X: Matrix):
    return 1 - torch.tanh(X) ** 2


def Sigmoid(X: Matrix):
    return torch.sigmoid(X)


def Sigmoid_gradient(X: Matrix):
    return Sigmoid(X) * (1 - Sigmoid(X))


def Srelu(al, tl, ar, tr):
    return lambda X: torch.where(X <= tl, tl + al * (X - tl),
                     torch.where(X < tr, X, tr + ar * (X - tr)))


def Srelu_gradient(al, tl, ar, tr):
    return lambda X: torch.where(X <= tl, al,
                     torch.where(X < tr, 1, ar))


class ActivationFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        raise NotImplementedError

    def gradient(self, X: Matrix) -> Matrix:
        raise NotImplementedError


class ReLUActivation(ActivationFunction):
    def __call__(self, X: Matrix) -> Matrix:
        return Relu(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Relu_gradient(X)


class LeakyReLUActivation(ActivationFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        return Leaky_relu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Leaky_relu_gradient(self.alpha)(X)


class AllReLUActivation(ActivationFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        return All_relu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        return All_relu_gradient(self.alpha)(X)


class HyperbolicTangentActivation(ActivationFunction):
    def __call__(self, X: Matrix) -> Matrix:
        return Hyperbolic_tangent(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Hyperbolic_tangent_gradient(X)


class SigmoidActivation(ActivationFunction):
    def __call__(self, X: Matrix) -> Matrix:
        return Sigmoid(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Sigmoid_gradient(X)


class SReLUActivation(ActivationFunction):
    def __init__(self, al=0.0, tl=0.0, ar=0.0, tr=1.0):
        # Store the parameters and their gradients in matrices.
        # This is to make them usable for optimizers.
        self.x = torch.Tensor([al, tl, ar, tr])
        self.Dx = torch.Tensor([0.0, 0.0, 0.0, 0.0])

    def __call__(self, X: Matrix) -> Matrix:
        al, tl, ar, tr = self.x
        return Srelu(al, tl, ar, tr)(X)

    def gradient(self, X: Matrix) -> Matrix:
        al, tl, ar, tr = self.x
        return Srelu_gradient(al, tl, ar, tr)(X)


def parse_activation(text: str) -> ActivationFunction:
    try:
        func = parse_function_call(text)
        if func.name == 'ReLU':
            return ReLUActivation()
        elif func.name == 'Sigmoid':
            return SigmoidActivation()
        elif func.name == 'HyperbolicTangent':
            return HyperbolicTangentActivation()
        elif func.name == 'AllReLU':
            alpha = func.as_scalar('alpha')
            return AllReLUActivation(alpha)
        elif func.name == 'LeakyReLU':
            alpha = func.as_scalar('alpha')
            return LeakyReLUActivation(alpha)
        elif func.name == 'SReLU':
            al = func.as_scalar('al', 0)
            tl = func.as_scalar('tl', 0)
            ar = func.as_scalar('ar', 0)
            tr = func.as_scalar('tr', 1)
            return SReLUActivation(al, tl, ar, tr)
    except:
        pass
    raise RuntimeError(f'Could not parse activation "{text}"')
