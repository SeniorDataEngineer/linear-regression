#!/usr/bin/env python3.8.5
# Copyright 2021, Rose Software Ltd, All rights reserved.

# Built-in imports.
from random import random

# Project imports.
from .mixin.has_property_mixin import HasPropertyMixin
from .neuron.artificial_neuron import Neuron
from .splitter.data_splitter import DataSplitter

# Third-party imports.
import numpy


class LinearRegressor(HasPropertyMixin):
    """
    This class can be used to perform linear regression for binary
    classification. It is a supervised machine learning algorithm.
    """

    def __init__(
            self,
            eta: float = 0.01,
            epochs: int = 100,
            neuron_count: int = 1,
            threshold: float = 0.0,
            **kwargs,):
        """
        Initializes a linear regressor. Takes eta which is the learning
        rate and epochs which is the number of simulations to run for a
        given target file path. It instantiates a DataSplitter for the
        specified target path. \n
        """
        self.eta = eta
        self.epochs = epochs
        self.threshold = threshold
        self.solutions = []

        self.splitter = DataSplitter(**kwargs)

        self._init_weights(4, True)

        kwargs = {
            'eta': self.eta,
        }
        self._init_neurons(neuron_count, **kwargs)

    def train_model(self):
        """
        Fits the model.
        """
        # Get features and actuals.
        X = [ # method returning numpy
            numpy.array(f[0: 4]) # hardcoded
            for f in self.splitter.get_training_samples_array() ]
        y = [
            y[-1] # hardcoded
            for y in self.splitter.get_training_samples_array() ]

        neuron = self.neurons[0] # weights should be in the neuron
        w = self.weights
        t = self.threshold

        for epoch in range(0, self.epochs):
            errors = 0
            for i in range(0, len(X)):
                # Calculate decision; delta.
                delta = self.eta * (
                    y[i] - neuron.decide_bias(w=w[1:], x=X[i], t=w[0]))
                
                # A pply delta to weights and update bias.
                w[1:] = w[1:] + delta * X[i]
                w[0] += delta

                # Tally erroneous decisions.
                errors += int(delta) != 0
            self.solutions.append({
                                'epoch': epoch,
                                'errors': errors, 
                                'error rate': errors/len(X),
                                'weights': w,
                                'function': f'f(x*T({w})'})
            if errors == 0:
                break

    def test_model(
            self,
            key: str) -> (numpy.array, numpy.array):
        """
        Tests the most accurate function against the testing samples
        and returns the results.
        """
        neuron = self.neurons[0]
        w = self.solutions[self.get_solution_index('error rate')].get('weights')

        self.splitter.set_field(key, numpy.nan)

        X = [
            numpy.array(f[0: 4]) # hardcoded
            for f in self.splitter.get_testing_samples_array() ]
        y = [
            y[-3:] # hardcoded
            for y in self.splitter.get_testing_samples_array() ]

        for i in range(0, len(X)):
            y[i][2] = neuron.decide_bias(w=w[1:], x=X[i], t=w[0])

        return (w, y)



    def _init_weights(
            self,
            n: int,
            bias: bool,):
        """
        Initialize n random weights plus a bias weight if specified.
        Store as numpy.array in Perceptron.
        """
        #features = self.splitter.get_feature_count() + bias - feat_count   # review this
        #samples = len(self.splitter.get_training_samples()) # remove this

        self.weights = [
                        random()
                        for _ in range(0, bias+n) ]

        self.weights[0] *= -1

        self.weights = numpy.array(self.weights)

    def _init_neurons(
            self,
            n: int,
            **kwargs,):
        """
        Initialize n Neurons.
        """
        self.neurons = [
                        Neuron(**kwargs)
                        for _ in range(0, n) ]

    def get_solution_index(
            self,
            key: str) -> int:
        """
        Iterate over the self.solutions and return the solution index
        with the lowest error rate.
        """
        max_ = float("-inf")
        index = None
        for i, solution in enumerate(self.solutions):
            if solution.get(key) > max_:
                index = i
        return index
