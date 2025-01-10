import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import _NDARRAY, FuncParamClass, check_min_len, convert_param_class

__all__ = ["ExpDecayingCos"]

_T = _t.TypeVar("_T")


class ExpDecayingCosParam(_t.Generic[_T], FuncParamClass):
    """Exponential Decaying Cosine parameters.

    Attributes:
        amplitude0 (float):
            The absolute amplitude of the decaying cosine.
        frequency (float):
            The frequency of the decaying cosine.
        phi0 (float):
            The initial phase of the decaying cosine.
        offset (float):
            The offset of the decaying cosine.
        tau (float):
            The decay constant of the decaying cosine.
        std (Optional[ExpDecayingCosParam]):
            The standard deviation of the parameters, if any.

    Additional attributes:
        omega (float):
            Calculates the angular frequency based on the frequency.
        rate (float):
            Calculates the rate of decay.
    """

    keys = ("amplitude0", "frequency", "phi0", "offset", "tau")

    amplitude0: _T
    frequency: _T
    phi0: _T
    offset: _T
    tau: _T

    @property
    def omega(self) -> _T:
        return 2 * np.pi * self.frequency  # pylint: disable=E1101  # type: ignore

    @property
    def rate(self) -> _T:
        return -1 / self.tau  # pylint: disable=E1101  # type: ignore


class ExpDecayingCosResult(ExpDecayingCosParam, FitResult[ExpDecayingCosParam]):
    param_class = convert_param_class(ExpDecayingCosParam)


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array(
        [abs(x[0]), x[1], (x[2] + (np.pi if x[0] < 0 else 0)) % (2 * np.pi), x[3], x[4]]
    )


def exp_decaying_cos_func(
    x: _NDARRAY,
    amplitude0: float,
    frequency: float,
    phi0: float,
    offset: float,
    tau: float,
):
    return (
        amplitude0 * np.cos(2 * np.pi * x * frequency + phi0) * np.exp(-x / tau)
        + offset
    )


def exp_decaying_cos_guess(x: _NDARRAY, y: _NDARRAY, **kwargs):
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
            The amplitude0 guess for the curve.
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
    tau: float = (max(x) - min(x)) / 5

    return np.array(
        normalize_res_list([sign_ * amp_guess, freq_guess, phase, off_guess, tau])
    )


class ExpDecayingCos(FitLogic[ExpDecayingCosResult]):  # type: ignore
    r"""Fit ExpDecayingCos function.


     Function
     ---------

     $$
     f(x) = A_0 * \cos(ω⋅x + \phi_0) * \exp(-x / τ) + A_{\text{offset}}
     $$

         f(x) = (
             amplitude0 * np.exp(-x / tau)
             * np.cos(2 * np.pi * x * frequency + phi0)
             + offset
         )


     Final parameters
    -----------------
     The final parameters are given by [`ExpDecayingCosParam`](../exp_decaying_cos_param/) dataclass.

    """

    _result_class: _t.Type[ExpDecayingCosResult] = ExpDecayingCosResult

    func = staticmethod(exp_decaying_cos_func)
    # func_std = staticmethod(cos_error)

    normalize_res = staticmethod(normalize_res_list)
    _guess = staticmethod(exp_decaying_cos_guess)

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        amplitude0: float = None,  # type: ignore
        frequency: float = None,  # type: ignore
        phi0: float = None,  # type: ignore
        offset: float = None,  # type: ignore
        tau: float = None,  # type: ignore
    ) -> "ExpDecayingCos": ...

    @classmethod
    def mask(cls, **kwargs) -> "ExpDecayingCos":
        return super().mask(**kwargs)

    _range_x = (0, np.inf)
