#!/usr/bin/env python3

# Copyright 2023 - 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path

from nerva_torch.activation_functions import ReLUActivation
from nerva_torch.datasets import create_npz_dataloaders, DataLoader
from nerva_torch.layers import ActivationLayer, LinearLayer
from nerva_torch.learning_rate import ConstantScheduler, LearningRateScheduler
from nerva_torch.loss_functions import SoftmaxCrossEntropyLossFunction, LossFunction
from nerva_torch.multilayer_perceptron import MultilayerPerceptron
from nerva_torch.training import compute_statistics
from nerva_torch.utilities import StopWatch


def sgd(M: MultilayerPerceptron,
        epochs: int,
        loss: LossFunction,
        learning_rate: LearningRateScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader
       ):

    training_time = 0.0

    for epoch in range(epochs):
        timer = StopWatch()
        lr = learning_rate(epoch)

        for (X, T) in train_loader:
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / Y.shape[0]
            M.backpropagate(Y, DY)
            M.optimize(lr)

        seconds = timer.seconds()
        training_time += seconds
        compute_statistics(M, lr, loss, train_loader, test_loader, epoch=epoch + 1, elapsed_seconds=seconds)

    print(f'Total training time for the {epochs} epochs: {training_time:.8f}s\n')


def main():
    if not Path("../data/mnist-flattened.npz").exists():
        print("Error: MNIST dataset not found. Please provide the correct location or run the prepare_datasets.py script first.")
        return

    train_loader, test_loader = create_npz_dataloaders("../data/mnist-flattened.npz", batch_size=100)

    M = MultilayerPerceptron()
    M.layers = [ActivationLayer(784, 1024, ReLUActivation()),
                ActivationLayer(1024, 512, ReLUActivation()),
                LinearLayer(512, 10)]
    for layer in M.layers:
        layer.set_optimizer('Momentum(0.9)')
        layer.set_weights('Xavier')

    loss = SoftmaxCrossEntropyLossFunction()
    learning_rate = ConstantScheduler(0.01)
    epochs = 10

    sgd(M, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    main()
