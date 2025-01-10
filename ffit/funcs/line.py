import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import _NDARRAY, FuncParamClass, check_min_len, convert_param_class

__all__ = ["Line"]

_T = _t.TypeVar("_T")


class LineParam(_t.Generic[_T], FuncParamClass):
    """Line parameters.

    Attributes:
        offset (float)
        amplitude (float)
    """

    keys = ("offset", "amplitude")

    offset: _T
    amplitude: _T


class LineResult(LineParam, FitResult[LineParam]):
    param_class = convert_param_class(LineParam)


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


class Line(FitLogic[LineResult]):  # type: ignore
    r"""Line function.
    ---------

    $$
    f(x) = a_0 + a_1 * x
    $$

        f(x) = offset + amplitude * x

    Final parameters:
    -----------------
    The final parameters are given by [`LineParam`](../line_param/) dataclass.


    """

    _result_class: _t.Type[LineResult] = LineResult
    func = staticmethod(line_func)
    _guess = staticmethod(line_guess)

    _example_param = (1, 3)

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        offset: float = None,  # type: ignore
        amplitude: float = None,  # type: ignore
    ) -> "Line": ...

    @classmethod
    def mask(cls, **kwargs) -> "Line":
        return super().mask(**kwargs)
