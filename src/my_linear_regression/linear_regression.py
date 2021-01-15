#!/usr/bin/env python3.8.5
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
from math import log

# Project imports.
from has_property_mixin import HasPropertyMixin


class LinearRegression(HasPropertyMixin):
    """
    This class can be used to perform linear regression for binary
    classification. It is a supervised machine learning algorithm.
    """

    