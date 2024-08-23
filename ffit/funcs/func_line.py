import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _NDARRAY, ParamDataclass, check_min_len


@dataclass(frozen=True)
class LineParam(ParamDataclass):
    offset: float
    amplitude: float
    std: "_t.Optional[LineParam]" = None


def line_func(x: _NDARRAY, offset: float, amplitude: float) -> _NDARRAY:
    return offset + amplitude * x


def line_guess(x: _NDARRAY, y: _NDARRAY, **kwargs):
    """Guess for line function.

    Args:
        x (_NDARRAY): x data.
        y (_NDARRAY): y data.

    Returns:
        _NDARRAY: (amplitude).
    """
    del kwargs
    if not check_min_len(x, y, 2):
        return np.ones(2)

    y = np.sort(y)
    x = np.sort(x)
    average_size = max(len(y) // 10, 1)
    y1 = np.average(y[:average_size])
    y2 = np.average(y[-average_size:])

    amplitude = (y2 - y1) / (x[-1] - x[0]) if x[-1] != x[0] else 1
    offset = y1 - x[0] * amplitude

    return np.array([offset, amplitude])


class Line(FitLogic[LineParam]):  # type: ignore
    r"""Fit Hyperbola function.


    Function:
    ---------

    $$
    f(x) = a_0 + a_1 * x
    $$

        f(x) = offset + amplitude * x

    Example:
    ---------
        >>> import ffit as ff
        >>> res = ff.Line.fit(x, y).res

        >>> res = ff.Line.fit(x, y, guess=[1, 2, 3, 4]).plot(ax).res
        >>> amplitude = res.amplitude

    Final parameters:
    -----------------
    - `offset`: float.
    - `amplitude`: float.

    """

    param: _t.Type[LineParam] = LineParam
    func = line_func
    _guess = line_guess
