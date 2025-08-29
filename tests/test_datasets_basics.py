import os
import tempfile
import unittest
import numpy as np
import torch

from utilities import to_tensor, all_close, check_tensors_are_close
from nerva_torch.datasets import to_one_hot, from_one_hot, MemoryDataLoader, infer_num_classes, create_npz_dataloaders


class TestDatasetsBasics(unittest.TestCase):
    def test_one_hot_roundtrip(self):
        idx = torch.tensor([0, 2, 1, 2], dtype=torch.long)
        oh = to_one_hot(idx, num_classes=3)
        back = from_one_hot(oh)
        self.assertTrue(torch.equal(back, idx))

    def test_memory_dataloader_batches_and_shapes(self):
        X = torch.randn(10, 4)
        T = torch.tensor([0, 1, 0, 2, 1, 2, 0, 1, 2, 2], dtype=torch.long)
        loader = MemoryDataLoader(X, T, batch_size=3)  # num_classes inferred -> 3
        batches = list(iter(loader))
        # floor(10/3) = 3 batches
        self.assertEqual(len(batches), 3)
        for Xi, Ti in batches:
            self.assertEqual(Xi.shape[0], 3)
            # one-hot
            self.assertEqual(Ti.shape[1], 3)

    def test_infer_num_classes_indices_vs_onehot(self):
        Ttrain = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        Ttest = torch.tensor([2, 0, 1, 1], dtype=torch.long)
        self.assertEqual(infer_num_classes(Ttrain, Ttest), 3)
        # one-hot
        Ttrain_oh = torch.eye(4)[:3]  # 3x4 one-hot with width 4
        self.assertEqual(infer_num_classes(Ttrain_oh, Ttest), 4)

    def test_create_npz_dataloaders_roundtrip(self):
        # Tiny dataset with 6 samples
        Xtrain = np.random.randn(6, 3).astype(np.float32)
        Ttrain = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
        Xtest = np.random.randn(6, 3).astype(np.float32)
        Ttest = np.array([1, 0, 2, 2, 0, 1], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tmp_dataset.npz")
            np.savez(path, Xtrain=Xtrain, Ttrain=Ttrain, Xtest=Xtest, Ttest=Ttest)
            train_loader, test_loader = create_npz_dataloaders(path, batch_size=2)
            # loaders produce one-hot of expected width
            for Xb, Tb in train_loader:
                self.assertEqual(Xb.shape[1], 3)
                self.assertEqual(Tb.shape[1], 3)
            for Xb, Tb in test_loader:
                self.assertEqual(Xb.shape[1], 3)
                self.assertEqual(Tb.shape[1], 3)


if __name__ == '__main__':
    unittest.main()
