import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _NDARRAY, ParamDataclass, check_min_len


@dataclass(frozen=True)
class HyperbolaParam(ParamDataclass):
    """Hyperbola parameters.

    Attributes:
        semix (float):
            The semi-major axis length of the hyperbola.
        semiy (float):
            The semi-minor axis length of the hyperbola.
        x0 (float):
            The x-coordinate of the hyperbola's center.
        y0 (float):
            The y-coordinate of the hyperbola's center.
        std (Optional[HyperbolaParam]):
            The standard deviation of the hyperbola parameters.
    """

    semix: float
    semiy: float
    x0: float
    y0: float
    std: "_t.Optional[HyperbolaParam]" = None


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
        direction = (
            1
            if np.mean(smoth_y[:average_size]) > np.mean(smoth_y[-average_size:])
            else -1
        )

    x0 = x[np.argmax(y)] if direction > 0 else x[np.argmin(y)]
    y0 = np.max(y) if direction > 0 else np.min(y)

    return np.array([np.std(x), -np.std(y) * direction, x0, y0])


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array([abs(x[0]), x[1], x[2], x[3]])


class Hyperbola(FitLogic[HyperbolaParam]):  # type: ignore
    r"""Hyperbola function.
    ---------

    $$
    \frac{(x - x0)^2}{semix^2} - \frac{(y - y0)^2}{semiy^2} = 1
    $$

        f(x) = y0 + semiy * np.sqrt(1 + ((x - x0) / semix) ** 2)

    Final parameters
    -----------------
    The final parameters are given by [`HyperbolaParam`](../hyperbola_param/) dataclass.

    """

    param: _t.Type[HyperbolaParam] = HyperbolaParam

    func = staticmethod(hyperbola_func)
    _guess = staticmethod(hyperbola_guess)
    normalize_res = staticmethod(normalize_res_list)

    _example_param = (1, 2, 0, 0.5)
    _example_x_min = -5
    _example_x_max = 5
