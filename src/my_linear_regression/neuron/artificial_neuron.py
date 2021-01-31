#!/usr/bin/env python3.8.5
# Copyright 2021, Rose Software Ltd, All rights reserved.

# Third party imports.
import numpy

# Project imports.
from ..mixin.has_property_mixin import HasPropertyMixin


class Neuron(HasPropertyMixin):
    """
    An artificial neuron as described by McCulloch-Pitts (MCP).
    Takes input and fires if threshold has been met.
    """

    def __init__(
            self,
            algorithm: object = "_linear_algebra",
            **kwargs,):
        """
        Initliazes a neuron with a particular algorithm for
        deciding whether to fire. The default algorithm is
        _linear_algebra.
        """
        if not self.has_object(name=algorithm):
            raise ValueError(f'The method {algorithm} does not exist on class'
                             f'{self.__class__}.')
        self.algorithm = getattr(self, algorithm)

    def set_report(
            self,
            report: dict):
        """
        Store the report for the Neuron function.
        """
        self.report = report

    def decide(
            self,
            w: numpy.ndarray,
            x: numpy.ndarray,
            t: float) -> float:
        """
        Decide, takes vector x and weight w and calculates
        whether to fire a -1 or 1; -1 indicates not firing and
        1 indicates firing. \n
        Returns:
            float
        Doctest:
            >>> n = Neuron()
            >>> assert n.decide([1,1,0], [0.25,0.3,0.1], 0.5) == 1
        """
        if self.algorithm(w, x) >= t:
            return 1
        return -1

    def decide_bias(
            self,
            w: numpy.ndarray,
            x: numpy.ndarray) -> float:
        """
        decide_bias uses the bias unit to subtract the threshold
        from the algorithm result. It is functionally the same as
        decide. \n
        Returns:
            float
        Doctest:
            >>> n = Neuron()
            >>> assert n.decide_bias([-1,1,1,0], [0.25,0.3,0.1], 0.5) == 1
        """
        if self.algorithm(w[1:], x) + w[0] >= 0:
            return 1
        return -1

    def _linear_algebra(
            self,
            w: numpy.ndarray,
            x: numpy.ndarray) -> float:
        """
        _linear_algebra calculates the sum of the products between
        w and x. \n
        Returns:
            float
        Doctest:
            >>> n = Neuron()
            >>> assert n._linear_algebra([1,2,3], [2,2,2]) == 12
        """
        if not len(w) == len(x):
            raise ValueError('Neuron ._linear_algebra: Length of w'
                             ' and x must be the same')
        return sum([
                    w[i] * x[i]
                    for i in range(0, len(w))])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
