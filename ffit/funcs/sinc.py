import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import _NDARRAY, FuncParamClass, check_min_len, convert_param_class

__all__ = ["Sinc"]

_T = _t.TypeVar("_T")
_FLOAT_EPS = np.finfo(float).eps


class SincParam(_t.Generic[_T], FuncParamClass):
    """Sinc function parameters.

    Attributes:
        amplitude (float):
            The amplitude of the sinc function.
        frequency (float):
            The frequency in 1/[x] units.
        x0 (float):
            The center position of the sinc function.
        offset (float):
            The global offset.

    Additional attributes:
        omega (float):
            The angular frequency.
    """

    keys = ("amplitude", "frequency", "x0", "offset")

    amplitude: _T
    frequency: _T
    x0: _T
    offset: _T

    @property
    def omega(self) -> _T:
        return 2 * np.pi * self.frequency  # pylint: disable=E1101  # type: ignore


class SincResult(SincParam, FitResult[SincParam]):
    param_class = convert_param_class(SincParam)


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    """Normalize the fit parameters.

    Rules:
    1. Amplitude is positive. Any sign is transferred to phase
    2. x0 is unchanged
    3. Array is not unpacked
    """
    return np.array([x[0], x[1], x[2], x[3]])


def sinc_func(x: _NDARRAY, amplitude: float, frequency: float, x0: float, offset: float):
    """Sinc function implementation.

    f(x) = amplitude * sin(2π * frequency * (x - x0)) / (2π * frequency * (x - x0)) + offset
    """
    return amplitude * np.sinc(frequency * (x - x0)) + offset


def sinc_sq_func(x: _NDARRAY, amplitude: float, frequency: float, x0: float, offset: float):
    """Sinc function implementation.

    f(x) = amplitude * sin(2π * frequency * (x - x0)) / (2π * frequency * (x - x0)) + offset
    """
    return amplitude * np.sinc(frequency * (x - x0)) ** 2 + offset


def sinc_guess(x: _NDARRAY, y: _NDARRAY, **kwargs):
    """Guess the initial parameters for fitting a sinc curve to the given data."""
    del kwargs
    if not check_min_len(x, y, 3):
        return np.zeros(4)

    sorted_y = np.sort(y)
    off_guess = np.mean(sorted_y[: len(sorted_y) // 4])
    amp_guess = np.mean(sorted_y[-len(sorted_y) // 4 :]) - off_guess
    freq_guess = 10.0 / (np.max(x) - np.min(x))
    x0_guess = x[np.argmax(np.abs(y - off_guess))]

    return np.array(normalize_res_list([amp_guess, freq_guess, x0_guess, off_guess]))  # type: ignore


class SincSq(FitLogic[SincResult]):  # type: ignore
    _result_class: _t.Type[SincResult] = SincResult

    func = staticmethod(sinc_sq_func)
    normalize_res = staticmethod(normalize_res_list)
    _guess = staticmethod(sinc_guess)

    _example_param = (1, 1, 0, 0)

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        amplitude: float = None,  # type: ignore
        frequency: float = None,  # type: ignore
        x0: float = None,  # type: ignore
        offset: float = None,  # type: ignore
    ) -> "SincSq": ...

    @classmethod
    def mask(cls, **kwargs) -> "SincSq":
        return super().mask(**kwargs)


class Sinc(FitLogic[SincResult]):  # type: ignore
    r"""Sinc function.
    --------

    $$
    f(x) = A \frac{\sin(ω(x-x_0))}{ω(x-x_0)} + A_0
    $$

        f(x) = amplitude * np.sinc(frequency * (x - x0)) + offset

    Alternative functions
    ----------------------

    `Sinc.SincSq`:

    $$
    f(x) = A \left(\frac{\sin(ω(x-x_0))}{ω(x-x_0)}\right)^2 + A_0
    $$

        f(x) = amplitude * np.sinc(frequency * (x - x0))**2 + offset


    Final parameters
    -----------------
    The final parameters are given by [`SincParam`](../sinc_param/) dataclass.
    """

    _result_class: _t.Type[SincResult] = SincResult

    func = staticmethod(sinc_func)
    normalize_res = staticmethod(normalize_res_list)
    _guess = staticmethod(sinc_guess)

    _example_param = (1, 1, 0, 0)

    SincSq = SincSq

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        amplitude: float = None,  # type: ignore
        frequency: float = None,  # type: ignore
        x0: float = None,  # type: ignore
        offset: float = None,  # type: ignore
    ) -> "Sinc": ...

    @classmethod
    def mask(cls, **kwargs) -> "Sinc":
        return super().mask(**kwargs)
