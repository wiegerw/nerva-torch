#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import sympy as sp

from nerva_sympy.loss_functions import LossFunction
from nerva_sympy.matrix_operations import elements_sum

Matrix = sp.Matrix


def matrix(name: str, rows: int, columns: int) -> Matrix:
    return Matrix(sp.symarray(name, (rows, columns), real=True))


def to_matrix(x):
    return sp.Matrix([[x]])


def equal_matrices(A: Matrix, B: Matrix) -> bool:
    m, n = A.shape
    return A.shape == B.shape and sp.simplify(A - B) == sp.zeros(m, n)


def sign(A: Matrix) -> Matrix:
    """
    Applies the sympy.sign function element-wise to a SymPy Matrix
    using nested list comprehensions.

    Args:
        A (Matrix): The input SymPy Matrix.

    Returns:
        Matrix: A new SymPy Matrix with the sign function applied to each element.
    """
    rows, cols = A.shape
    elements = [[sp.sign(A[i, j]) for j in range(cols)] for i in range(rows)]
    return Matrix(elements)


class AbsoluteErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return elements_sum(abs(Y - T))

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return sign(Y - T)


def test_loss_function_gradient():
    """Validates the gradient of AbsoluteErrorLossFunction."""
    K = 3
    N = 2

    loss = AbsoluteErrorLossFunction()

    # vector
    y = matrix('y', 1, K)
    t = matrix('t', 1, K)
    y_value = to_matrix(loss(y, t))
    Dy = to_matrix(loss.gradient(y, t))
    y_jacobian = y_value.jacobian(y)
    assert equal_matrices(Dy, y_jacobian)

    # matrix
    Y = matrix('Y', N, K)
    T = matrix('T', N, K)
    DY = to_matrix(loss.gradient(Y, T))
    for i in range(N):
        Dy_i = to_matrix(loss.gradient(Y.row(i), T.row(i)))
        assert equal_matrices(Dy_i, DY.row(i))


if __name__ == '__main__':
    test_loss_function_gradient()