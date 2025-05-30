#!/usr/bin/env python3

# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import torch

from nerva_torch.loss_functions import Squared_error_loss, Softmax_cross_entropy_loss
from nerva_torch.loss_functions_torch import squared_error_loss_torch, softmax_cross_entropy_loss_torch
from utilities import random_float_matrix, make_target


class TestLossFunction(TestCase):
    def test_loss(self):
        a = 0.0001
        b = 1.0
        shape = (3, 4)
        for i in range(100):
            Y = random_float_matrix(shape, a, b)
            T = make_target(Y).astype(float)

            Y = torch.Tensor(Y)
            T = torch.Tensor(T)

            loss_ms1 = squared_error_loss_torch(Y, T)
            loss_ms2 = Squared_error_loss(Y, T)
            self.assertAlmostEqual(loss_ms1, loss_ms2, delta=1e-5)

            loss_sc1 = softmax_cross_entropy_loss_torch(Y, T)
            loss_sc2 = Softmax_cross_entropy_loss(Y, T)
            self.assertAlmostEqual(loss_sc1, loss_sc2, delta=1e-5)


if __name__ == '__main__':
    import unittest
    unittest.main()