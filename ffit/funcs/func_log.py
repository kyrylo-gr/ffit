import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _NDARRAY, ParamDataclass, check_min_len


@dataclass(frozen=True)
class LogParam(ParamDataclass):
    amplitude: float
    rate: float
    offset: float

    std: "_t.Optional[LogParam]" = None

    def amplitude_at_base(self, base: float = 10):
        return self.amplitude / np.log(base)

    def offset_at_base(self, base: float = 10):
        return self.offset + self.amplitude * np.log(self.rate) * (1 / np.log(base) - 1)


def ln_func(x, amplitude, rate, offset):
    return amplitude * np.log(rate * x) + offset - amplitude * np.log(rate)


def log_guess(x: _NDARRAY, y: _NDARRAY, **kwargs):
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

    # y1 = a * ln(b * x1) + c
    # y2 = a * ln(b * x2) + c
    # y3 = a * ln(b * x3) + c
    #
    # y1 - y2 = a * (ln(b * x1) - ln(b * x2))
    # y3 - y2 = a * (ln(b * x3) - ln(b * x2))
    #
    # (y1 - y2) / (y3 - y2) = (ln(b * x1) - ln(b * x2)) / (ln(b * x3) - ln(b * x2))
    # ln(b*x) ≈ b*x -1
    #
    # (y1-y2)/(y3-y2) =

    # concave = 1 if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0 else -1
    b = abs((y2 - y1) / (y3 - y2) * (x3 - x2) / (x2 - x1) / 3)
    z1 = np.log(b * x1)
    z3 = np.log(b * x3)
    # print(y1, y3, z1, z3, b)
    # y1 = a * log(b * x1) + c = a * z1 + c
    # y3 = a * log(b * x3) + c = a * z3 + c
    #
    # => a = (y1 - y3) / (z1 - z3)
    # => c = (y1 * z3 - y3 * z1) / (z3 - z1)

    a = (y1 - y3) / (z1 - z3)
    c = (y1 * z3 - y3 * z1) / (z3 - z1) + a * np.log(b)
    return np.array([a, b, c])


class Log(FitLogic[LogParam]):  # type: ignore
    r"""Fit Log function.


    Function
    ---------
    $$
    f(x) = A * \ln(b*x)) + A_0 - A * \ln(b)
    $$

        f(x) = amplitude * np.log(rate * x) + offset - amplitude * np.log(rate)

    Random base
    ------------

    For function with the random base of the logarithm:
    $$
    f(x) = A * \log_d(b*x) + A_0 - A * \ln(b)
    $$

    We can rewrite it as:

    $$
    f(x) = \frac{A}{\ln(d)} * \ln(b*x) + A_0 - \frac{A}{\ln(d)} * \ln(b)
    $$

    Therefore, not natural base can be rewritten as amplitude renormalization.
    You can use `amplitude_at_base` method to get the right amplitude.

    Example
    ---------
        >>> import ffit as ff
        >>> res = ff.Log.fit(x, y).res

        >>> res = ff.Log.fit(x, y, guess=[1, 2, 3]).plot(ax).res
        >>> amplitude = res.amplitude

    Final parameters
    -----------------
    - `amplitude`: float.
        The amplitude of the logarithm function.
    - `rate`: float.
        The rate of the logarithm function.
    - `offset`: float.
        The offset at x=1.
    - `amplitude_at_base(base: float = 10)`: float.
        The amplitude if the base is not natural. (A * ln(d)).
    - `offset_at_base(base: float = 10)`: float.
        The offset if the base is not natural.
        $ A_0 + A * \ln(b) * (1 / \ln(d) - 1) $
    """

    param: _t.Type[LogParam] = LogParam

    func = ln_func
    _guess = log_guess
