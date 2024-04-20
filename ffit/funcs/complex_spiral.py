import typing as _t

import numpy as np

from ..fit_logic import FitLogic


class ComplexSpiralParam(_t.NamedTuple):
    amplitude0: float
    phi0: float
    freq: float
    tau: float
    offset: float


def complex_spiral_func(x, amplitude0, phi0, freq, tau, offset):  # pylint: disable=W0221
    ampl = amplitude0 * np.exp(1j * phi0)
    return ampl * np.exp(1j * freq * 2 * np.pi * x - x / tau) + offset + 1j * offset


class ComplexSpiral(FitLogic[ComplexSpiralParam]):
    param: _t.Type[ComplexSpiralParam] = ComplexSpiralParam

    @staticmethod
    def func(x, amplitude0, phi0, freq, tau, offset):  # pylint: disable=W0221
        ampl = amplitude0 * np.exp(1j * phi0)
        return ampl * np.exp(1j * freq * 2 * np.pi * x - x / tau) + offset + 1j * offset

    @staticmethod
    def _guess(x, z, **kwargs):  # pylint: disable=W0237
        the_fft = np.fft.fft(z - z.mean())
        index_max = np.argmax(np.abs(the_fft))
        freq = np.fft.fftfreq(len(z), d=x[1] - x[0])[index_max]
        ampl = the_fft[index_max]

        return [
            (np.max(np.real(z)) - np.min(np.real(z))) / 2,
            np.angle(ampl),
            freq,
            np.max(x) / 2,
            (np.max(np.real(z)) + np.min(np.real(z))) / 2,
        ]
