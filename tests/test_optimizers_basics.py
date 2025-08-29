import unittest
import torch

from utilities import to_tensor, all_close, check_tensors_are_close
from nerva_torch.optimizers import (
    GradientDescentOptimizer,
    MomentumOptimizer,
    NesterovOptimizer,
    CompositeOptimizer,
    parse_optimizer,
)


class TestOptimizersBasics(unittest.TestCase):
    def test_parse_optimizer_valid(self):
        self.assertTrue(callable(parse_optimizer("GradientDescent")))
        self.assertTrue(callable(parse_optimizer("Momentum(mu=0.9)")))
        self.assertTrue(callable(parse_optimizer("Nesterov(mu=0.9)")))

    def test_parse_optimizer_invalid(self):
        with self.assertRaises(RuntimeError):
            parse_optimizer("Unknown()")

    def test_gradient_descent_update(self):
        x = torch.tensor([1.0, -2.0])
        Dx = torch.tensor([0.5, -1.0])
        opt = GradientDescentOptimizer(x, Dx)
        opt.update(eta=0.2)
        self.assertTrue(torch.allclose(x, torch.tensor([1.0 - 0.1, -2.0 + 0.2])))

    def test_momentum_update(self):
        x = torch.tensor([0.0, 0.0])
        Dx = torch.tensor([1.0, -2.0])
        opt = MomentumOptimizer(x, Dx, mu=0.9)
        opt.update(eta=0.1)  # delta_x = 0.9*0 - 0.1*Dx = [-0.1, 0.2]
        self.assertTrue(torch.allclose(x, torch.tensor([-0.1, 0.2])))
        # Update again with same gradient
        opt.update(eta=0.1)  # delta_x = 0.9*[-0.1,0.2] - 0.1*Dx = [-0.19, 0.38]; x += delta_x
        self.assertTrue(torch.allclose(x, torch.tensor([-0.29, 0.58])))

    def test_nesterov_update(self):
        x = torch.tensor([0.0, 0.0])
        Dx = torch.tensor([1.0, -2.0])
        opt = NesterovOptimizer(x, Dx, mu=0.9)
        opt.update(eta=0.1)
        # delta_x = 0.9*0 - 0.1*Dx = [-0.1, 0.2]; x += 0.9*delta_x - 0.1*Dx = [-0.19, 0.38]
        self.assertTrue(torch.allclose(x, torch.tensor([-0.19, 0.38])))

    def test_composite_optimizer(self):
        x1 = torch.tensor([1.0])
        Dx1 = torch.tensor([2.0])
        x2 = torch.tensor([3.0])
        Dx2 = torch.tensor([4.0])
        o1 = GradientDescentOptimizer(x1, Dx1)
        o2 = GradientDescentOptimizer(x2, Dx2)
        comp = CompositeOptimizer([o1, o2])
        comp.update(eta=0.5)
        self.assertTrue(torch.allclose(x1, torch.tensor([1.0 - 1.0])))
        self.assertTrue(torch.allclose(x2, torch.tensor([3.0 - 2.0])))


if __name__ == '__main__':
    unittest.main()
