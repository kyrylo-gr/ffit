import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _NDARRAY, ParamDataclass


@dataclass(frozen=True)
class ComplexSpiralParam(ParamDataclass):
    amplitude0: float
    frequency: float
    phi0: float
    tau: float
    offset_amp: float
    offset_phase: float
    std: "_t.Optional[ComplexSpiralParam]" = None

    def offset(self):
        return self.offset_amp * np.exp(1j * self.offset_phase)

    def amplitude(self):
        return self.amplitude0 * np.exp(1j * self.phi0)


def complex_spiral_func(
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


class ComplexSpiral(FitLogic[ComplexSpiralParam]):
    param: _t.Type[ComplexSpiralParam] = ComplexSpiralParam
    func = staticmethod(complex_spiral_func)

    @staticmethod
    def _guess(x, z, **kwargs):  # pylint: disable=W0237
        the_fft = np.fft.fft(z - z.mean())
        index_max = np.argmax(np.abs(the_fft))
        freq = np.fft.fftfreq(len(z), d=x[1] - x[0])[index_max]
        ampl = the_fft[index_max]

        return [
            (np.max(np.real(z)) - np.min(np.real(z))) / 2,
            freq,
            np.angle(ampl),
            np.max(x) / 2,
            (np.max(np.real(z)) + np.min(np.real(z))) / 2,
            0,
        ]

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
    ) -> "ComplexSpiral": ...

    @classmethod
    def mask(cls, **kwargs) -> "ComplexSpiral":
        return super().mask(**kwargs)
