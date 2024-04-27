import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _NDARRAY, check_min_len


class ExpParam(_t.NamedTuple):
    amplitude: float
    rate: float
    offset: float

    @property
    def tau(self):
        return -1 / self.rate


def exp_func(x, amplitude, rate, offset):
    return amplitude * np.exp(rate * x) + offset


def exp_guess(x: _NDARRAY, y: _NDARRAY, **kwargs):
    del kwargs
    if not check_min_len(x, y, 3):
        return np.ones(3)
    average_size = max(len(y) // 10, 1)

    data = np.array([x, y]).T
    sorted_data = data[data[:, 1].argsort()]

    x1 = np.mean(sorted_data[:average_size, 0])
    y1 = np.mean(sorted_data[:average_size, 1])
    x2 = np.mean(sorted_data[average_size:-average_size, 0])
    y2 = np.mean(sorted_data[average_size:-average_size, 1])
    x3 = np.mean(sorted_data[-average_size:, 0])
    y3 = np.mean(sorted_data[-average_size:, 1])
    if x1 == x2 or x2 == x3 or x1 == x3:  # noqa
        return np.ones(3)
    if y1 == y2 or y2 == y3 or y1 == y3:  # noqa
        return np.ones(3)
    #  y1 = a * exp(b * x1) + c
    #  y2 = a * exp(b * x2) + c
    #  y3 = a * exp(b * x3) + c
    # y1 - y2 = a * (exp(b * x1) - exp(b * x2))
    # y3 - y2 = a * (exp(b * x3) - exp(b * x2))
    # (y1 - y2) / (y3 - y2) = (exp(b * x1) - exp(b * x2)) / (exp(b * x3) - exp(b * x2))
    # exp(b*x) ≈ 1 + b*x
    # (y1-y2)/(y3-y2) =
    concave = 1 if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0 else -1
    b = concave * (y2 - y1) / (y3 - y2) * (x3 - x2) / (x2 - x1) / 3
    z1 = np.exp(b * x1)
    z3 = np.exp(b * x3)
    #  y1 = a * exp(b * x1) + c = a * z1 + c
    #  y3 = a * exp(b * x3) + c = a * z3 + c
    # => a = (y1 - y3) / (z1 - z3)
    # => c = (y1 * z3 - y3 * z1) / (z3 - z1)
    a = (y1 - y3) / (z1 - z3)
    c = (y1 * z3 - y3 * z1) / (z3 - z1)
    return np.array([a, b, c])
    # offset = y1 if concave > 0 else y3
    # amplitude = concave * (y3 - y1) * np.sign(x3 - x1)

    # print(amplitude, concave)
    # print(amplitude, concave / abs(x3 - x1), offset)
    # print((x1, y1), (x2, y2), (x3, y3))
    # return amplitude, concave / abs(x3 - x1), offset  # - amplitude * np.exp(x1)


class Exp(FitLogic[ExpParam]):
    r"""Fit Exp function.


    Function
    ---------

    $$
    f(x) = A * \exp(b*x) + A_0
    $$

        f(x) = amplitude * np.exp(rate * x) + offset

    Example
    ---------
        >>> import ffit as ff
        >>> res = ff.Exp.fit(x, y).res

        >>> res = ff.Exp.fit(x, y, guess=[1, 2, 3]).plot(ax).res
        >>> amplitude = res.amplitude

    Final parameters
    -----------------
    - `amplitude`: float.
        The amplitude of the exponential function.
    - `rate`: float.
        The rate of the exponential function.
    - `offset`: float.
        The offset of the exponential function.
    - `tau`: float.
        The constant of the damping exponential. ($\tau$ = -1 / rate)

    """

    param: _t.Type[ExpParam] = ExpParam

    func = exp_func
    _guess = exp_guess
