# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)


from typing import List

from nerva_torch.datasets import DataLoader, create_npz_dataloaders
from nerva_torch.learning_rate import LearningRateScheduler, parse_learning_rate
from nerva_torch.loss_functions import parse_loss_function, LossFunction
from nerva_torch.multilayer_perceptron import MultilayerPerceptron, parse_multilayer_perceptron
from nerva_torch.utilities import StopWatch, pp, set_numpy_options, set_torch_options


class SGDOptions(object):
    debug=False


def print_epoch(epoch, lr, loss, train_accuracy, test_accuracy, elapsed):
    print(f'epoch {epoch:3}  '
          f'lr: {lr:.8f}  '
          f'loss: {loss:.8f}  '
          f'train accuracy: {train_accuracy:.8f}  '
          f'test accuracy: {test_accuracy:.8f}  '
          f'time: {elapsed:.8f}s'
         )

def compute_accuracy(M: MultilayerPerceptron, data_loader: DataLoader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=1)  # the predicted classes for the batch
        targets = T.argmax(axis=1)    # the expected classes
        total_correct += (predicted == targets).sum().item()

    return total_correct / N


def compute_loss(M: MultilayerPerceptron, data_loader: DataLoader, loss: LossFunction):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        Y = M.feedforward(X)
        total_loss += loss(Y, T)

    return total_loss / N


def compute_statistics(M, lr, loss, train_loader, test_loader, epoch, elapsed_seconds=0.0, print_statistics=True):
    if print_statistics:
        train_loss = compute_loss(M, train_loader, loss)
        train_accuracy = compute_accuracy(M, train_loader)
        test_accuracy = compute_accuracy(M, test_loader)
        print_epoch(epoch, lr, train_loss, train_accuracy, test_accuracy, elapsed_seconds)
    else:
        print(f'epoch {epoch:3}')


def sgd(M: MultilayerPerceptron,
        epochs: int,
        loss: LossFunction,
        learning_rate: LearningRateScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader
       ):

    lr = learning_rate(0)
    compute_statistics(M, lr, loss, train_loader, test_loader, epoch=0)
    training_time = 0.0

    for epoch in range(epochs):
        timer = StopWatch()
        lr = learning_rate(epoch)  # update the learning at the start of each epoch

        for k, (X, T) in enumerate(train_loader):
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / Y.shape[0]

            if SGDOptions.debug:
                print(f'epoch: {epoch} batch: {k}')
                M.info()
                pp("X", X)
                pp("Y", Y)
                pp("DY", DY)

            M.backpropagate(Y, DY)
            M.optimize(lr)

        seconds = timer.seconds()
        training_time += seconds
        compute_statistics(M, lr, loss, train_loader, test_loader, epoch=epoch + 1, elapsed_seconds=seconds)

    print(f'Total training time for the {epochs} epochs: {training_time:.8f}s\n')


def train(layer_specifications: List[str],
          linear_layer_sizes: List[int],
          linear_layer_optimizers: List[str],
          linear_layer_weight_initializers: List[str],
          batch_size: int,
          epochs: int,
          loss: str,
          learning_rate: str,
          weights_and_bias_file: str,
          dataset_file: str,
          debug: bool
         ):
    SGDOptions.debug = debug
    set_numpy_options()
    set_torch_options()
    loss = parse_loss_function(loss)
    learning_rate = parse_learning_rate(learning_rate)
    M = parse_multilayer_perceptron(layer_specifications, linear_layer_sizes, linear_layer_optimizers, linear_layer_weight_initializers)
    if weights_and_bias_file:
        M.load_weights_and_bias(weights_and_bias_file)
    train_loader, test_loader = create_npz_dataloaders(dataset_file, batch_size=batch_size)
    sgd(M, epochs, loss, learning_rate, train_loader, test_loader)
