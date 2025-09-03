# Copyright 2022 - 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import unittest
from nerva_torch.loss_functions import SoftmaxCrossEntropyLossFunction, LossFunction, SquaredErrorLossFunction, \
    NegativeLogLikelihoodLossFunction, CrossEntropyLossFunction, LogisticCrossEntropyLossFunction
from nerva_torch.loss_functions import Squared_error_loss, Softmax_cross_entropy_loss
from nerva_torch.loss_functions_torch import squared_error_loss_torch, softmax_cross_entropy_loss_torch
from utilities import to_tensor, random_float_matrix, make_target, as_float


class TestLossFunctions(unittest.TestCase):
    def test_loss1(self):
        Y = to_tensor([
            [0.23759169, 0.42272727, 0.33968104],
            [0.43770149, 0.28115265, 0.28114586],
            [0.20141643, 0.45190243, 0.34668113],
            [0.35686849, 0.17944701, 0.46368450],
            [0.48552814, 0.26116029, 0.25331157],
        ])

        T = to_tensor([
            [1.00000000, 0.00000000, 0.00000000],
            [1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000],
            [1.00000000, 0.00000000, 0.00000000],
        ])

        # List of (loss function, expected value)
        losses = [
            (SquaredErrorLossFunction(), 2.6550281475767563),
            (SoftmaxCrossEntropyLossFunction(), 5.106889686512423),
            (NegativeLogLikelihoodLossFunction(), 4.548777728936653),
            (CrossEntropyLossFunction(), 4.548777728936653),
            (LogisticCrossEntropyLossFunction(), 2.539463487358204),
        ]

        for loss_fn, expected in losses:
            L = as_float(loss_fn(Y, T))
            self.assertAlmostEqual(L, expected, places=5, msg=f"{loss_fn.__class__.__name__} failed: got {L}, expected {expected}")

    def test_loss2(self):
        Y = to_tensor([
            [0.24335898, 0.40191852, 0.35472250],
            [0.21134093, 0.53408849, 0.25457058],
            [0.24788846, 0.42021140, 0.33190014],
            [0.40312318, 0.24051313, 0.35636369],
            [0.43329234, 0.34433141, 0.22237625],
        ])

        T = to_tensor([
            [1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [1.00000000, 0.00000000, 0.00000000],
        ])

        # List of (loss function, expected value)
        losses = [
            (SquaredErrorLossFunction(), 3.6087104890568256),
            (SoftmaxCrossEntropyLossFunction(), 5.5889911807479065),
            (NegativeLogLikelihoodLossFunction(), 5.90971538007391),
            (CrossEntropyLossFunction(), 5.909715380073911),
            (LogisticCrossEntropyLossFunction(), 2.7376380548462254),
        ]

        for loss_fn, expected in losses:
            L = as_float(loss_fn(Y, T))
            self.assertAlmostEqual(L, expected, places=5, msg=f"{loss_fn.__class__.__name__} failed: got {L}, expected {expected}")

    def test_loss3(self):
        Y = to_tensor([
            [0.23774258, 0.42741216, 0.33484526],
            [0.29687977, 0.43115409, 0.27196615],
            [0.43420442, 0.22655227, 0.33924331],
            [0.28599538, 0.35224692, 0.36175770],
            [0.20014798, 0.43868708, 0.36116494],
        ])

        T = to_tensor([
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000],
        ])

        # List of (loss function, expected value)
        losses = [
            (SquaredErrorLossFunction(), 3.289394384977318),
            (SoftmaxCrossEntropyLossFunction(), 5.441938177932827),
            (NegativeLogLikelihoodLossFunction(), 5.44627595910772),
            (CrossEntropyLossFunction(), 5.44627595910772),
            (LogisticCrossEntropyLossFunction(), 2.678127590042374),
        ]

        for loss_fn, expected in losses:
            L = as_float(loss_fn(Y, T))
            self.assertAlmostEqual(L, expected, places=5, msg=f"{loss_fn.__class__.__name__} failed: got {L}, expected {expected}")


    def test_loss4(self):
        Y = to_tensor([
            [0.26787616, 0.35447135, 0.37765249],
            [0.26073833, 0.45527664, 0.28398503],
            [0.31560020, 0.41003295, 0.27436685],
            [0.37231605, 0.17984538, 0.44783858],
            [0.49308039, 0.27786731, 0.22905230],
        ])

        T = to_tensor([
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000],
            [1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000],
            [0.00000000, 0.00000000, 1.00000000],
        ])

        # List of (loss function, expected value)
        losses = [
            (SquaredErrorLossFunction(), 3.521376994732803),
            (SoftmaxCrossEntropyLossFunction(), 5.548304798627446),
            (NegativeLogLikelihoodLossFunction(), 5.726367921857207),
            (CrossEntropyLossFunction(), 5.726367921857208),
            (LogisticCrossEntropyLossFunction(), 2.7197402348335156),
        ]

        for loss_fn, expected in losses:
            L = as_float(loss_fn(Y, T))
            self.assertAlmostEqual(L, expected, places=5, msg=f"{loss_fn.__class__.__name__} failed: got {L}, expected {expected}")

    def test_loss5(self):
        Y = to_tensor([
            [0.29207765, 0.40236525, 0.30555710],
            [0.38987005, 0.36536339, 0.24476656],
            [0.24441444, 0.32191037, 0.43367519],
            [0.38397493, 0.35636403, 0.25966104],
            [0.29902507, 0.25018760, 0.45078733],
        ])

        T = to_tensor([
            [0.00000000, 0.00000000, 1.00000000],
            [1.00000000, 0.00000000, 0.00000000],
            [1.00000000, 0.00000000, 0.00000000],
            [0.00000000, 1.00000000, 0.00000000],
            [0.00000000, 0.00000000, 1.00000000],
        ])

        # List of (loss function, expected value)
        losses = [
            (SquaredErrorLossFunction(), 3.2404999669186503),
            (SoftmaxCrossEntropyLossFunction(), 5.4240756991825645),
            (NegativeLogLikelihoodLossFunction(), 5.365012502539291),
            (CrossEntropyLossFunction(), 5.365012502539292),
            (LogisticCrossEntropyLossFunction(), 2.6711745146065176),
        ]

        for loss_fn, expected in losses:
            L = as_float(loss_fn(Y, T))
            self.assertAlmostEqual(L, expected, places=5, msg=f"{loss_fn.__class__.__name__} failed: got {L}, expected {expected}")


class CompareLossFunctions(unittest.TestCase):
    def test_loss(self):
        a = 0.0001
        b = 1.0
        shape = (3, 4)
        for i in range(100):
            Y = random_float_matrix(shape, a, b)
            T = make_target(Y).astype(float)

            Y = to_tensor(Y)
            T = to_tensor(T)

            loss_ms1 = squared_error_loss_torch(Y, T)
            loss_ms2 = Squared_error_loss(Y, T)
            self.assertAlmostEqual(loss_ms1, loss_ms2, delta=1e-5)

            loss_sc1 = softmax_cross_entropy_loss_torch(Y, T)
            loss_sc2 = Softmax_cross_entropy_loss(Y, T)
            self.assertAlmostEqual(loss_sc1, loss_sc2, delta=1e-5)
