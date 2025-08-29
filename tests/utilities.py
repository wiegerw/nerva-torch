# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import torch


def random_float_matrix(shape, a, b):
    """
    Generates a random numpy array with the given shape and float values in the range [a, b].

    Parameters:
    shape (tuple): The shape of the numpy array to generate.
    a (float): The minimum value in the range.
    b (float): The maximum value in the range.

    Returns:
    np.ndarray: A numpy array of the specified shape with random float values in the range [a, b].
    """
    # Generate a random array with values in the range [0, 1)
    rand_array = np.random.rand(*shape)

    # Scale and shift the array to the range [a, b]
    scaled_array = a + (b - a) * rand_array

    return scaled_array


# --- Framework-agnostic helpers for tests ---
# For other backends (TensorFlow/JAX/NumPy), reimplement these functions
# with identical signatures and semantics.

def to_tensor(array):
    """Convert a Python list or NumPy array to the backend tensor.

    For PyTorch, returns torch.float32 by default for floating arrays; uses long for integer arrays.
    """
    if isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.integer):
        return torch.tensor(array, dtype=torch.long)
    # fall back to float tensor
    return torch.tensor(array, dtype=torch.float32)


def all_close(X1, X2, atol=1e-6, rtol=1e-6):
    """Backend-agnostic allclose check."""
    return torch.allclose(X1, X2, atol=atol, rtol=rtol)


def check_tensors_are_close(name1, X1, name2, X2, atol=1e-6, rtol=1e-6):
    """Assert that two tensors are close, with helpful diagnostics."""
    if not all_close(X1, X2, atol=atol, rtol=rtol):
        diff = torch.abs(X1 - X2)
        max_diff = torch.max(diff).item()
        raise AssertionError(f"Tensors {name1} and {name2} are not close. Max diff: {max_diff:.8f}")


def make_target(Y: np.ndarray) -> np.ndarray:
    """
    Creates a boolean matrix T with the same shape as Y,
    where each row of T has exactly one value set to 1.

    Parameters:
    Y (np.ndarray): The input numpy array.

    Returns:
    np.ndarray: A boolean matrix with the same shape as Y,
                with exactly one True value per row.
    """
    if Y.ndim != 2:
        raise ValueError("The input array must be two-dimensional")

    # Get the shape of the input array
    rows, cols = Y.shape

    # Initialize an array of zeros with the same shape as Y
    T = np.zeros((rows, cols), dtype=bool)

    # Set one random element in each row to True
    for i in range(rows):
        random_index = np.random.randint(0, cols)
        T[i, random_index] = True

    return T
