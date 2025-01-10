import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import _NDARRAY, FuncParamClass, check_min_len, convert_param_class

__all__ = ["Cos"]

_T = _t.TypeVar("_T")


class CosParam(_t.Generic[_T], FuncParamClass):
    """Cos function parameters.

    Attributes:
    -----------------
        amplitude (float):
            The amplitude.
        frequency (float):
            The frequency in 1/[x] units.
        phi0 (float):
            The phase inside cos.
        offset (float):
            The global offset.

    Additional attributes:
    -----------------
        omega (float):
            The angular frequency.

    """

    # __slots__ = ("amplitude", "frequency", "phi0", "offset")
    keys = ("amplitude", "frequency", "phi0", "offset")
    amplitude: _T
    frequency: _T
    phi0: _T
    offset: _T

    @property
    def omega(self) -> _T:
        return 2 * np.pi * self.frequency  # pylint: disable=E1101  # type: ignore


class CosResult(CosParam, FitResult[CosParam]):
    param_class = convert_param_class(CosParam)


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array(
        [abs(x[0]), x[1], (x[2] + (np.pi if x[0] < 0 else 0)) % (2 * np.pi), x[3]]
    )


def cos_func(
    x: _NDARRAY, amplitude: float, frequency: float, phi0: float, offset: float
):
    return amplitude * np.cos(2 * np.pi * x * frequency + phi0) + offset


def cos_guess(x: _NDARRAY, y: _NDARRAY, **kwargs):
    """Guess the initial parameters for fitting a curve to the given data.

    Parameters:
    - x: array-like
        The x-coordinates of the data points.
    - y: array-like
        The y-coordinates of the data points.
    - **kwargs: keyword arguments
        Additional arguments that can be passed to the function.

    Returns:
    - list
        A list containing the initial parameter guesses for fitting the curve.
        The list contains the following elements:
        - sign_ * amp_guess: float
            The amplitude guess for the curve.
        - period: float
            The period guess for the curve.
        - off_guess: float
            The offset guess for the curve.
    """
    del kwargs
    if not check_min_len(x, y, 3):
        return np.zeros(4)

    off_guess: float = np.mean(y)  # type: ignore
    amp_guess: float = np.abs(np.max(y - off_guess))
    nnn = 10 * len(y)
    fft_vals = np.fft.rfft(y - off_guess, n=nnn)
    fft_freqs = np.fft.rfftfreq(nnn, d=x[1] - x[0])
    freq_max_index = np.argmax(np.abs(fft_vals))
    freq_guess: float = np.abs(fft_freqs[freq_max_index])
    sign_: float = np.sign(np.real(fft_vals[freq_max_index]))  # type: ignore
    phase: float = np.imag(fft_vals[freq_max_index])

    return np.array(
        normalize_res_list([sign_ * amp_guess, freq_guess, phase, off_guess])
    )


class Cos(FitLogic[CosResult]):  # type: ignore
    r"""Cosine function.
    --------

    $$
    f(x) = A * \cos(ω ⋅ x + \phi_0) + A_0
    $$

        f(x) = amplitude * cos(2 * pi * frequency * x + phi0) + offset

    Final parameters
    -----------------
    The final parameters are given by [`CosParam`](../cos_param/) dataclass.
    """

    _result_class: _t.Type[CosResult] = CosResult

    func = staticmethod(cos_func)
    # func_std = staticmethod(cos_error)

    normalize_res = staticmethod(normalize_res_list)
    _guess = staticmethod(cos_guess)

    _example_param = (1, 1, 1.0, 1.0)

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        amplitude: float = None,  # type: ignore
        frequency: float = None,  # type: ignore
        phi0: float = None,  # type: ignore
        offset: float = None,  # type: ignore
    ) -> "Cos": ...

    @classmethod
    def mask(cls, **kwargs) -> "Cos":
        return super().mask(**kwargs)
