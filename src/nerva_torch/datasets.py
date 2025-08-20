# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""In-memory data loader helpers and one-hot conversions.

The DataLoader defined here mirrors a small subset of the PyTorch
DataLoader API but operates on in-memory tensors loaded from .npz files.
"""

from pathlib import Path
from typing import Tuple, Union

import torch
from nerva_torch.utilities import load_dict_from_npz


def to_one_hot(x: torch.LongTensor, num_classes: int) -> torch.Tensor:
    """Convert class index tensor to one-hot matrix with num_classes columns."""
    one_hot = torch.zeros(len(x), num_classes, dtype=torch.float)
    one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot


def from_one_hot(one_hot: torch.Tensor) -> torch.LongTensor:
    """Convert one-hot encoded rows to class index tensor."""
    return torch.argmax(one_hot, dim=1).long()


class MemoryDataLoader(object):
    """A minimal data loader with an interface similar to torch.utils.data.DataLoader."""

    def __init__(self, Xdata: torch.Tensor, Tdata: torch.LongTensor, batch_size: int=True, num_classes=0):
        """Iterate batches over row-major tensors; one-hot encode targets if needed.

        If Tdata is a vector of class indices and num_classes > 0 (or can be
        inferred), batches yield (X, one_hot(T)). Otherwise targets are returned as-is.
        """
        self.Xdata = Xdata
        self.Tdata = Tdata
        self.batch_size = batch_size
        self.dataset = Xdata
        self.num_classes = int(Tdata.max() + 1) if num_classes == 0 and len(Tdata.shape) == 1 else num_classes

    def __iter__(self):
        N = self.Xdata.shape[0]  # N is the number of examples
        K = N // self.batch_size  # K is the number of batches
        for k in range(K):
            batch = range(k * self.batch_size, (k + 1) * self.batch_size)
            yield self.Xdata[batch], to_one_hot(self.Tdata[batch], self.num_classes) if self.num_classes else self.Tdata[batch]

    def __len__(self):
        """Number of batches."""
        return self.Xdata.shape[0] // self.batch_size


DataLoader = Union[MemoryDataLoader, torch.utils.data.DataLoader]


def create_npz_dataloaders(filename: str, batch_size: int=True) -> Tuple[MemoryDataLoader, MemoryDataLoader]:
    """Creates a data loader from a file containing a dictionary with Xtrain, Ttrain, Xtest and Ttest tensors."""
    path = Path(filename)
    print(f'Loading dataset from file {path}')
    if not path.exists():
        raise RuntimeError(f"Could not load file '{path}'")

    data = load_dict_from_npz(filename)
    Xtrain, Ttrain, Xtest, Ttest = data['Xtrain'], data['Ttrain'], data['Xtest'], data['Ttest']
    train_loader = MemoryDataLoader(Xtrain, Ttrain, batch_size)
    test_loader = MemoryDataLoader(Xtest, Ttest, batch_size)
    return train_loader, test_loader