import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..utils import check_min_len


class HyperbolaParam(_t.NamedTuple):
    semix: float
    semiy: float
    x0: float
    y0: float


def hyperbola_func(x, semix, semiy, x0, y0):
    return y0 + semiy * np.sqrt(1 + ((x - x0) / semix) ** 2)


def hyperbola_guess(x, y, **kwargs):
    if not check_min_len(x, y, 3):
        return np.zeros(4)
    direction = kwargs.get("direction")

    if direction is None:
        average_size = max(len(y) // 10, 1)
        smoth_y = np.convolve(y, np.ones(average_size) / average_size, mode="valid")
        smoth_y = np.diff(smoth_y)
        direction = 1 if np.mean(smoth_y[:average_size]) > np.mean(smoth_y[-average_size:]) else -1

    x0 = x[np.argmax(y)] if direction > 0 else x[np.argmin(y)]
    y0 = np.max(y) if direction > 0 else np.min(y)

    return HyperbolaParam(
        semix=np.std(x),  # type: ignore
        semiy=-np.std(y) * direction,  # type: ignore
        x0=x0,
        y0=y0,  # np.mean(y),
    )


def normalize_res_list(x: _t.Sequence[float]) -> list:
    return [abs(x[0]), x[1], x[2], x[3]]


class Hyperbola(FitLogic[HyperbolaParam]):
    r"""Fit Hyperbola function.


    Function
    ---------

    $$
    \frac{(x - x0)^2}{semix^2} - \frac{(y - y0)^2}{semiy^2} = 1
    $$

        f(x) = y0 + semiy * np.sqrt(1 + ((x - x0) / semix) ** 2)

    Example
    -------
        >>> import ffit as ff
        >>> res = ff.Hyperbola.fit(x, y).res

        >>> res = ff.Hyperbola.fit(x, y, guess=[1, 2, 3, 4]).plot(ax).res
        >>> semix = res.semix

    Final parameters
    -----------------
    - `semix`: float.
        The semi-x axis of the hyperbola.
    - `semiy`: float.
        The semi-y axis of the hyperbola.
    - `x0`: float.
        The origin of the x.
    - `y0`: float.
        The origin of the y.
    """

    param: _t.Type[HyperbolaParam] = HyperbolaParam

    func = hyperbola_func
    _guess = hyperbola_guess
    normalize_res = normalize_res_list
