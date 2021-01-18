#!/usr/bin/env python3.8.5
# Copyright 2021, Rose Software Ltd, All rights reserved.

# Built-in imports.
from math import log

# Project imports.
from has_property_mixin import HasPropertyMixin


class LinearRegression(HasPropertyMixin):
    """
    This class can be used to perform linear regression for binary
    classification. It is a supervised machine learning algorithm.
    """

    def __init__(
            self,
            eta: float = 0.01,
            epochs: int = 100,
            target: str)
        """
        Initializes a linear regressor. Takes eta which is the learning
        rate and epochs which is the number of simulations to run for a
        given target file path. \n
        """
        self.eta = eta
        self.epochs = epochs
        self.target = target
