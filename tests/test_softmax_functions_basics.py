import unittest
import torch

from nerva_torch import softmax_functions as sm
from utilities import to_tensor, all_close, check_tensors_are_close


class TestSoftmaxFunctionsBasics(unittest.TestCase):
    def test_softmax_rowwise_properties(self):
        X = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        Y = sm.softmax(X)
        # rows sum to 1
        self.assertTrue(torch.allclose(Y.sum(dim=1), torch.ones(X.shape[0]), atol=1e-6))
        # positive entries
        self.assertTrue((Y > 0).all())

    def test_softmax_invariance_to_shift(self):
        X = torch.randn(4, 5)
        c = torch.randn(4, 1)
        Y1 = sm.softmax(X)
        Y2 = sm.softmax(X + c)  # adding per-row constant shouldn't change result
        self.assertTrue(torch.allclose(Y1, Y2, atol=1e-6))

    def test_stable_softmax_matches_on_moderate_values(self):
        X = torch.randn(3, 4)
        self.assertTrue(torch.allclose(sm.softmax(X), sm.stable_softmax(X), atol=1e-6))

    def test_log_softmax_relationship(self):
        X = torch.randn(3, 4)
        Y = sm.softmax(X)
        LS = sm.log_softmax(X)
        self.assertTrue(torch.allclose(torch.exp(LS), Y, atol=1e-6))

    def test_stable_log_softmax_large_values(self):
        X = torch.tensor([[1000.0, 1001.0, 1002.0]])
        # Should remain finite
        LS = sm.stable_log_softmax(X)
        self.assertTrue(torch.isfinite(LS).all())
        # Stable and naive should agree when subtracting max
        Xs = X - X.max(dim=1, keepdim=True).values
        self.assertTrue(torch.allclose(sm.log_softmax(Xs), sm.stable_log_softmax(X), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
