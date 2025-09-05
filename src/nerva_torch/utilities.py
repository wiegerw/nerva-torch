# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Miscellaneous utilities (formatting, timing, parsing, I/O)."""

import re
import time
from typing import Dict, Union

import numpy as np
import torch


def set_numpy_options():
    """Configure NumPy print options for readable output."""
    np.set_printoptions(precision=8, edgeitems=3, threshold=5, suppress=True, linewidth=160)


def set_torch_options():
    """Configure PyTorch print options and multiprocessing strategy."""
    torch.set_printoptions(precision=8, edgeitems=3, threshold=5, sci_mode=False, linewidth=160)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')


def pp_numpy(name: str, arr: np.ndarray):
    """Internal helper: pretty-print using NumPy arrays only."""
    if arr.ndim == 1:
        print(f"{name} ({arr.shape[0]})")
    elif arr.ndim == 2:
        print(f"{name} ({arr.shape[0]}x{arr.shape[1]})")
    else:
        print(f"{name} {arr.shape}")
    print(arr)


def pp(name: str, x: torch.Tensor):
    """Pretty-print a tensor with name and shape info, using NumPy formatting."""
    pp_numpy(name, x.detach().cpu().numpy())


class StopWatch(object):
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        """Get elapsed time in seconds since creation or last reset."""
        end = time.perf_counter()
        return end - self.start

    def reset(self):
        """Reset the timer to the current time."""
        self.start = time.perf_counter()


class FunctionCall:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments

    def has_key(self, key: str) -> bool:
        """Check if the given key exists in parsed arguments."""
        return key in self.arguments

    def get_value(self, key: str) -> str:
        if key in self.arguments:
            return self.arguments[key]
        elif len(self.arguments) == 1 and "" in self.arguments:
            return self.arguments[""]
        return ""

    def as_scalar(self, key: str, default_value: float = None) -> float:
        value = self.get_value(key)
        if value:
            return float(value)  # Assuming scalar is a float, adjust if necessary
        elif default_value:
            return default_value
        raise RuntimeError(f"Could not find an argument named \"{key}\"")

    def as_string(self, key: str, default_value: str = "") -> str:
        value = self.get_value(key)
        if value:
            return value
        elif default_value:
            return default_value
        raise RuntimeError(f"Could not find an argument named \"{key}\"")


def parse_function_call(text: str) -> FunctionCall:
    """
    Parse a string of the shape `NAME(key1=value1, key2=value2, ...)`.
    If there are no arguments the parentheses may be omitted.
    If there is only one parameter, it is allowed to pass `NAME(value)` instead of `NAME(key=value)`
    """

    text = text.strip()

    def error():
        raise RuntimeError(f"Could not parse function call \"{text}\"")

    name = ""
    arguments = {}

    # no parentheses
    match = re.match(r"(\w+$)", text)
    if match:
        name = match.group(1)
        return FunctionCall(name, arguments)

    # with parentheses
    match = re.match(r"(\w+)\((.*?)\)", text)
    if match:
        name = match.group(1)
        args = re.split(r",", match.group(2))

        if len(args) == 1 and "=" not in args[0]:
            # NAME(value)
            value = args[0].strip()
            arguments[""] = value
            return FunctionCall(name, arguments)
        else:
            # NAME(key1=value1, ...)
            for arg in args:
                words = re.split(r"\s*=\s*", arg.strip())
                if len(words) != 2:
                    error()
                key, value = words
                if key in arguments:
                    print(f"Key \"{key}\" appears multiple times.")
                    error()
                arguments[key] = value
            return FunctionCall(name, arguments)

    error()


def load_dict_from_npz(filename: str) -> Dict[str, Union[torch.Tensor, torch.LongTensor]]:
    """Loads a dictionary from a file in .npz format"""
    def make_tensor(x: np.ndarray) -> Union[torch.Tensor, torch.LongTensor]:
        if np.issubdtype(x.dtype, np.integer):
            return torch.LongTensor(x)
        return torch.Tensor(x)

    data = dict(np.load(filename, allow_pickle=True))
    data = {key: make_tensor(value) for key, value in data.items()}
    return data


def save_dict_to_npz(filename: str, data: Dict[str, torch.Tensor]):
    """Saves a dictionary of torch tensors to a compressed .npz file."""
    if not filename.endswith(".npz"):
        filename += ".npz"

    # convert all torch tensors to numpy arrays
    numpy_data = {key: value.detach().cpu().numpy() for key, value in data.items()}
    np.savez_compressed(filename, **numpy_data)
