#!/usr/bin/env python3.8.5
# Copyright 2021, Rose Software Ltd, All rights reserved.

# Built-in imports.
from random import randint

# Project imports.
from .mixin.has_property_mixin import HasPropertyMixin
from .neuron.artificial_neuron import Neuron

# Third-party imports.
import numpy


class LinearRegressor(HasPropertyMixin):
    """
    This class can be used to perform linear regression for binary
    classification. It is a supervised machine learning algorithm.
    """

    def __init__(
            self,
            samples_train: numpy.ndarray,
            samples_test: numpy.ndarray,
            labels_train: numpy.ndarray,
            labels_test: numpy.ndarray,
            bias: bool = True,
            eta: float = 0.001,
            epochs: int = 50,):
        """
        Initializes a linear regressor. Takes eta which is the learning
        rate and epochs which is the number of simulations to run for a
        given target file path. It instantiates a DataSplitter for the
        specified target path. \n
        """
        self.eta = eta
        self.epochs = epochs
        self.samples_train = samples_train
        self.samples_test = samples_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.evolutions = []

    def set_weights(self, bias: bool):
        """
        Set the random weights, one for each feature that is used for
        training.
        """
        rgen = numpy.random.RandomState(1)
        self.weights = rgen.normal(
            loc=0.0,
            scale=0.01,
            size=bias+self.samples_train.shape[1])

    @DeprecationWarning
    def set_weights_(self):
        """
        Set the random weights, one for each feature that is used for
        training and the bias. \n
        """
        self.weights = [
                        randint(1, 1000) / 5000 * -1
                        for _ in range(0, len(self.samples_train[0])+1) ]

    def train_model(self):
        """
        Fits the model.
        """
        self.set_weights(bias=bias)
        w = self.weights
        X = self.samples_train
        Y = self.labels_train

        for epoch in range(0, self.epochs):
            neuron = Neuron()
            errors = 0
            for i in range(0, len(X)):
                # Calculate decision; delta.
                delta = self.eta * (
                    Y[i][-1] - neuron.decide_bias(w=w, x=X[i]))

                # Apply delta to weights and update bias.
                w[1:] += delta * X[i]
                w[0] += delta

                # Tally incorrect predictions.
                errors += int(delta != 0.0)

            # Set report and store neuron.
            neuron.set_report([epoch, errors/len(X), w])
            self.evolutions.append(neuron)

            if errors == 0:
                break

    def test_model(
            self,
            neuron: Neuron,) -> (list, list, list):
        """
        Tests a neuron-function against the testing samples and
        returns a tuple that includes 'class label as text', 'class
        int representation' and 'int prediction'. \n
        Returns:
            (numpy.array, numpy.array, numpy.array)
        """
        w = neuron.report[-1]
        X = self.samples_test
        Y = self.labels_test
        txt_label = []
        int_label = []
        pre_label = []
        for i in range(0, len(X)):
            txt_label.append(Y[i][0])
            int_label.append(Y[i][1])
            pre_label.append(neuron.decide_bias(w=w, x=X[i]))

        return (txt_label, int_label, pre_label)

    def test_model_(
            self,
            neuron: Neuron,) -> list:
        """
        Tests a function against the testing samples and returns
        a list of 'int prediction'. \n
        Returns:
            numpy.array
        """
        w = neuron.report[-1]
        X = self.samples_test
        Y = self.labels_test
        pre_label = []
        for i in range(0, len(X)):
            pre_label.append(neuron.decide_bias(w=w, x=X[i]))

        return pre_label

    def get_evolutions(self) -> int:
        """
        Return all evolutions of the neuron.
        """
        return self.evolutions


if __name__ == "__main__":
    import doctest
    doctest.testmod()
