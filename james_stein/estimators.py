"""A collection of relevant estimators to Stein's Paradox.

Assumed that the data has unit mean.
"""

import numpy as np


from numpy.linalg import norm
from .shao_strawderman import GenerateSS


def naive(x):
    """Return Naive Estimate."""
    return x


def js(x, p=None, a=None):
    """Return James-Stein Estimate."""
    if a is not None:
        num = a
    elif p is not None:
        num = p - 2
    else:
        raise ValueError("Need to provide a value for either a or p.")
    return (1 - num / (norm(x) ** 2)) * x


def js_ve(x, p):
    """Return Positive-Part James-Stein Estimate."""
    return np.maximum(0, js(x, p=p))


def generate_ss(p, p_star=None, a=None):
    """Return Shao-Strawderman Estimator.

    Example
    -------
    ss = generate_ss(5)
    estimate = ss(x)
    """
    return GenerateSS(p).estimator(p_star, a)
