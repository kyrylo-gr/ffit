import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import _NDARRAY, FuncParamClass, check_min_len, convert_param_class

__all__ = ["Hyperbola"]

_T = _t.TypeVar("_T")


class HyperbolaParam(_t.Generic[_T], FuncParamClass):
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

    keys = ("semix", "semiy", "x0", "y0")

    semix: _T
    semiy: _T
    x0: _T
    y0: _T


class HyperbolaResult(HyperbolaParam, FitResult[HyperbolaParam]):
    param_class = convert_param_class(HyperbolaParam)


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


class Hyperbola(FitLogic[HyperbolaResult]):  # type: ignore
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

    _result_class: _t.Type[HyperbolaResult] = HyperbolaResult

    func = staticmethod(hyperbola_func)
    _guess = staticmethod(hyperbola_guess)
    normalize_res = staticmethod(normalize_res_list)

    _example_param = (1, 2, 0, 0.5)
    _example_x_min = -5
    _example_x_max = 5

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        semix: float = None,  # type: ignore
        semiy: float = None,  # type: ignore
        x0: float = None,  # type: ignore
        y0: float = None,  # type: ignore
    ) -> "Hyperbola": ...

    @classmethod
    def mask(cls, **kwargs) -> "Hyperbola":
        return super().mask(**kwargs)
