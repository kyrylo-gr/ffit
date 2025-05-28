import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import FuncParamClass, check_min_len, convert_param_class

__all__ = ["Parabola"]

_T = _t.TypeVar("_T")


class ParabolaParam(_t.Generic[_T], FuncParamClass):
    """Parabola parameters.

    Attributes:
        amplitude (float):
            The amplitude of the parabola.
        x0 (float):
            The center of the parabola.
        y0 (float):
            The offset of the parabola.
        std (Optional[ParabolaParam]):
            The standard deviation of the parabola parameters.
    """

    keys = ("amplitude", "x0", "y0")

    amplitude: _T
    x0: _T
    y0: _T

    __latex_repr__ = r"$&amplitude \cdot (x - &x0)^2 + &y0$"
    __latex_repr_symbols__ = {
        "amplitude": r"A",
        "x0": r"x_0",
        "y0": r"y_0",
    }


class ParabolaResult(ParabolaParam, FitResult[ParabolaParam]):
    param_class = convert_param_class(ParabolaParam)


def parabola_func(x, amplitude, x0, y0):
    return amplitude * (x - x0) ** 2 + y0


def parabola_guess(x, y, **kwargs):
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

    return np.array([-np.std(y) * direction, x0, np.mean(y)])


class Parabola(FitLogic[ParabolaResult]):  # type: ignore
    r"""Parabola function.
    ---------

    $$
        y = A * (x - x_0) ** 2 + y_0
    $$

        f(x) = amplitude * (x - x0) ** 2 + y0

    Final parameters
    -----------------
    The final parameters are given by [`ParabolaParam`](../parabola_param/) dataclass.

    """

    _result_class: _t.Type[ParabolaResult] = ParabolaResult

    func = staticmethod(parabola_func)
    _guess = staticmethod(parabola_guess)

    _example_param = (1, 2, 0)
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
    ) -> "Parabola": ...

    @classmethod
    def mask(cls, **kwargs) -> "Parabola":
        return super().mask(**kwargs)
