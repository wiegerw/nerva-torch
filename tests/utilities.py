# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import math
from typing import Union, Sequence
import numpy as np
import torch

# ------------------------
# Tensor conversion
# ------------------------

def to_tensor(array: Union[Sequence, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert a Python list, NumPy array, or PyTorch tensor to a PyTorch tensor.
    - Float arrays become torch.float32.
    - Integer arrays become torch.long.
    - Torch tensors are returned as-is.
    """
    if isinstance(array, torch.Tensor):
        return array
    if isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.integer):
        return torch.tensor(array, dtype=torch.long)
    return torch.tensor(array, dtype=torch.float32)


def to_long(array: Union[Sequence, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert a Python list, NumPy array, or PyTorch tensor to torch.long."""
    if isinstance(array, torch.Tensor):
        return array.long()
    return torch.tensor(array, dtype=torch.long)


# ------------------------
# Tensor comparison
# ------------------------

def equal_tensors(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Check if two tensors are exactly equal."""
    return bool(torch.equal(x, y))


def almost_equal(a: Union[float, int, torch.Tensor],
                 b: Union[float, int, torch.Tensor],
                 rel_tol: float = 1e-5,
                 abs_tol: float = 1e-8) -> bool:
    """
    Compare two numeric scalars (float, int, or 0-d PyTorch tensor) approximately.
    Returns True if close within given relative and absolute tolerances.
    """
    # Extract scalar if tensor
    for x in (a, b):
        if isinstance(x, torch.Tensor):
            x = x.item()
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def all_close(X1: torch.Tensor, X2: torch.Tensor, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Compare two PyTorch tensors approximately. Returns True if all elements are close."""
    return torch.allclose(X1, X2, rtol=rtol, atol=atol)


def all_true(mask: torch.Tensor) -> bool:
    """Return True if all elements of a boolean tensor are True."""
    return bool(torch.all(mask).item())


def all_finite(x: torch.Tensor) -> bool:
    """Return True if all elements of a tensor are finite."""
    return bool(torch.isfinite(x).all().item())


def all_positive(X: torch.Tensor) -> bool:
    """Return True if all entries of X are strictly positive."""
    return (X > 0).all().item()


# ------------------------
# Random tensors
# ------------------------

def randn(*shape: int) -> torch.Tensor:
    """Return a random normal tensor of given shape."""
    return torch.randn(*shape)


def rand(*shape: int) -> torch.Tensor:
    """Return a uniform random tensor in [0,1) of given shape."""
    return torch.rand(*shape)


# ------------------------
# Test helpers
# ------------------------

def assert_tensors_are_close(name1: str, X1: torch.Tensor,
                             name2: str, X2: torch.Tensor,
                             rtol: float = 1e-6, atol: float = 1e-6):
    """
    Assert that two tensors are close, with helpful diagnostics.
    Raises AssertionError if not.
    """
    if not all_close(X1, X2, rtol=rtol, atol=atol):
        diff = torch.abs(X1 - X2)
        max_diff = torch.max(diff).item()
        raise AssertionError(f"Tensors {name1} and {name2} are not close. Max diff: {max_diff:.8f}")


def as_float(x: torch.Tensor) -> float:
    """Convert a 0-d Torch tensor to a Python float."""
    if x.ndim != 0:
        raise ValueError("Input must be 0-dimensional")
    return float(x.item())

# ------------------------
# Test generation
# ------------------------

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
