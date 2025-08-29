import unittest
import torch

from nerva_torch import matrix_operations as mo
from utilities import to_tensor, all_close, check_tensors_are_close


class TestMatrixOperationsBasics(unittest.TestCase):
    def test_zeros_ones_identity(self):
        Z = mo.zeros(2, 3)
        O = mo.ones(2, 3)
        I = mo.identity(3)
        self.assertEqual(Z.shape, (2, 3))
        self.assertTrue(torch.all(Z == 0))
        self.assertEqual(O.shape, (2, 3))
        self.assertTrue(torch.all(O == 1))
        self.assertEqual(I.shape, (3, 3))
        self.assertTrue(torch.allclose(I, torch.eye(3)))

    def test_product_and_hadamard(self):
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        Y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        self.assertTrue(torch.allclose(mo.product(X, Y), X @ Y))
        self.assertTrue(torch.allclose(mo.hadamard(X, Y), X * Y))

    def test_sums_means_max(self):
        X = torch.tensor([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
        self.assertTrue(torch.allclose(mo.columns_sum(X), torch.sum(X, dim=0)))
        self.assertTrue(torch.allclose(mo.rows_sum(X), torch.sum(X, dim=1)))
        self.assertTrue(torch.allclose(mo.columns_mean(X), torch.mean(X, dim=0)))
        self.assertTrue(torch.allclose(mo.rows_mean(X), torch.mean(X, dim=1)))
        self.assertTrue(torch.allclose(mo.columns_max(X), torch.max(X, dim=0).values))
        self.assertTrue(torch.allclose(mo.rows_max(X), torch.max(X, dim=1).values))

    def test_inv_sqrt_stability_and_log_sigmoid(self):
        X = torch.tensor([0.0, 1.0, 4.0])
        inv = mo.inv_sqrt(X)
        # Finite values due to epsilon
        self.assertTrue(torch.isfinite(inv).all())
        # Compare log_sigmoid to torch F.logsigmoid
        Y = torch.linspace(-10, 10, steps=11)
        self.assertTrue(torch.allclose(mo.log_sigmoid(Y), torch.nn.functional.logsigmoid(Y), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
