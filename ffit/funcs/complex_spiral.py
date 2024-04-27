import typing as _t

import numpy as np

from ..fit_logic import FitLogic


class ComplexSpiralParam(_t.NamedTuple):
    amplitude0: float
    phi0: float
    freq: float
    tau: float
    offset: float


def complex_spiral_func(x, params: np.ndarray):
    """Complex spiral function.

    Parameters:
    - 0: amplitude0,
    - 1: phi0
    - 2: freq
    - 3: tau
    - 4: offset
    # TODO: Add complex offset phase
    """
    ampl = params[0] * np.exp(1j * params[1])
    return ampl * np.exp(1j * params[2] * 2 * np.pi * x - x / params[3]) + params[4]


class ComplexSpiral(FitLogic[ComplexSpiralParam]):
    param: _t.Type[ComplexSpiralParam] = ComplexSpiralParam
    func = complex_spiral_func

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
