import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import _NDARRAY, EquationClass, FuncParamClass, convert_param_class

__all__ = ["ComplexSpiralExpDecay", "ComplexSpiralGaussianDecay", "ComplexSpiral"]

_T = _t.TypeVar("_T")


class ComplexSpiralParam(_t.Generic[_T], FuncParamClass):
    """Complex spiral parameters.

    Attributes:
        amplitude0 (float):
            The absolute amplitude of the spiral.
        frequency (float):
            The frequency of the spiral.
        phi0 (float):
            The phase of the spiral.
        tau (float):
            The time constant of the spiral.
        offset_amp (float):
            The amplitude offset of the spiral.
        offset_phase (float):
            The phase offset of the spiral.

    More attributes:

    - offset (complex):
            Calculates the complex offset based on the amplitude and phase offsets.
    - amplitude (complex):
            Calculates the complex amplitude based on the initial amplitude and phase.
    - rate (float):
            Calculates the rate of decay of the spiral.
    - omega (float):
            Calculates the angular frequency of the spiral.
    """

    keys = (
        "amplitude0",
        "frequency",
        "phi0",
        "tau",
        "offset_amp",
        "offset_phase",
    )

    amplitude0: _T
    frequency: _T
    phi0: _T
    tau: _T
    offset_amp: _T
    offset_phase: _T

    @property
    def offset(self) -> _T:
        return self.offset_amp * np.exp(1j * self.offset_phase)  # type: ignore

    @property
    def amplitude(self) -> _T:
        return self.amplitude0 * np.exp(1j * self.phi0)  # type: ignore

    @property
    def rate(self) -> _T:
        return -1.0 / self.tau  # type: ignore

    @property
    def omega(self) -> _T:
        return 2 * np.pi * self.frequency  # type: ignore


class ComplexSpiralResult(ComplexSpiralParam, FitResult[ComplexSpiralParam]):
    param_class = convert_param_class(ComplexSpiralParam)
    label: ComplexSpiralParam[EquationClass]  # type: ignore


def complex_spiral_exp_decay_func(
    x: _NDARRAY, amplitude0, frequency, phi0, tau, offset_amp, offset_phase
):
    """Complex spiral function.

    Parameters:
    - 0: amplitude0,
    - 2: frequency
    - 1: phi0
    - 3: tau
    - 4: offset amplitude
    - 5: offset phase

    """
    ampl = amplitude0 * np.exp(1j * phi0)
    return (
        ampl * np.exp(1j * frequency * 2 * np.pi * x - x / tau)
        + np.exp(1j * offset_phase) * offset_amp
    )


def complex_spiral_gaussian_decay_func(
    x: _NDARRAY, amplitude0, frequency, phi0, tau, offset_amp, offset_phase
):
    """Complex spiral function.

    Parameters:
    - 0: amplitude0,
    - 2: frequency
    - 1: phi0
    - 3: tau
    - 4: offset amplitude
    - 5: offset phase

    """
    amplitude = amplitude0 * np.exp(1j * phi0)
    return (
        amplitude * np.exp(1j * frequency * 2 * np.pi * x - x**2 / tau**2)
        + np.exp(1j * offset_phase) * offset_amp
    )


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array([x[0], x[1], x[2] % (2 * np.pi), x[3], x[4], x[5] % (2 * np.pi)])


class ComplexSpiralExpDecay(FitLogic[ComplexSpiralResult]):  # type: ignore
    _result_class: _t.Type[ComplexSpiralResult] = ComplexSpiralResult

    func = staticmethod(complex_spiral_exp_decay_func)
    normalize_res = staticmethod(normalize_res_list)

    @staticmethod
    def _guess(x, z, **kwargs):  # pylint: disable=W0237
        the_fft = np.fft.fft(z - z.mean())
        index_max = np.argmax(np.abs(the_fft))
        freq = np.fft.fftfreq(len(z), d=x[1] - x[0])[index_max]
        ampl = the_fft[index_max]

        return np.array(
            [
                (np.max(np.real(z)) - np.min(np.real(z))) / 2,
                freq,
                np.angle(ampl),
                np.max(x) / 2,
                (np.max(np.real(z)) + np.min(np.real(z))) / 2,
                0,
            ]
        )

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        amplitude: float = None,  # type: ignore
        frequency: float = None,  # type: ignore
        phi0: float = None,  # type: ignore
        tau: float = None,  # type: ignore
        offset_amp: float = None,  # type: ignore
        offset_phase: float = None,  # type: ignore
    ) -> "ComplexSpiralExpDecay": ...

    @classmethod
    def mask(cls, **kwargs) -> "ComplexSpiralExpDecay":
        return super().mask(**kwargs)

    _range_x = (0.0, np.inf)
    _doc_ignore = True


class ComplexSpiralGaussianDecay(ComplexSpiralExpDecay):  # type: ignore
    func = staticmethod(complex_spiral_gaussian_decay_func)
    _doc_ignore = True


class ComplexSpiral(ComplexSpiralExpDecay):
    r"""Complex Spiral function.
    --------
    By default, the function has exponential decay:

    $$
    f(x) = Z_0 * \exp(i⋅ω⋅x) \exp(-x/τ) + Z_{\text{offset}}
    $$

        f(x) = (
            amplitude0 * np.exp(1j * phi0)
            * np.exp(1j * 2 * np.pi * frequency * x - x / tau)
            + np.exp(1j * offset_phase) * offset_amp
        )

    Alternative functions
    ----------------------

    `ComplexSpiral.GaussianDecay`:

    $$
        f(x) = Z_0 \exp(i⋅ω⋅x) \exp(-x^2/τ^2) + Z_{\text{offset}}
    $$

        f(x) = (
            amplitude0 * np.exp(1j * phi0)
            * np.exp(1j * frequency * 2 * np.pi * x - x**2 / tau**2)
            + np.exp(1j * offset_phase) * offset_amp


    Final parameters
    -----------------
    The final parameters are given by [`ComplexSpiralParam`](../complex_spiral_param/) dataclass.
    """

    GaussianDecay = ComplexSpiralGaussianDecay
    ExpDecay = ComplexSpiralExpDecay
    _doc_ignore = False
