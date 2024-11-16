import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import ParamDataclass


@dataclass(frozen=True)
class LorentzParam(ParamDataclass):
    """Lorentz parameters.

    Attributes:
        f0 (float):
            The center frequency of the Lorentzian.
        amplitude (float):
            The amplitude of the Lorentzian.
        bandwidth (float):
            The bandwidth of the Lorentzian.
        phi0 (float):
            The phase at the center frequency.
        amplitude0 (float):
            The amplitude at the center frequency.
        delay (float):
            The electrical delay.
        amplitude_phase (float):
            The average phase of the impedance.
    """

    f0: float
    amplitude: float
    bandwidth: float
    phi0: float
    amplitude0: float
    delay: float
    amplitude_phase: float

    std: "_t.Optional[LorentzParam]" = None


def lorentz_func(freqs, f0, ampl, bandwidth, phi0, ampl0, delay, phaseampl):
    del delay
    orig = ampl0 * np.exp(1j * phi0)
    return (
        orig - np.exp(1j * phaseampl) * ampl / (1j * (freqs - f0) / (bandwidth) + 1)
    ) * 1


def lorentz_guess(x, z, **kwargs):
    """
    Estimate the parameters for fitting a model to the given frequency and impedance data.

    Parameters:
    - freqs (array-like): Array of frequency values.
    - zs (array-like): Array of impedance values.

    Returns:
    - f0 (float): Estimated center frequency.
    - ampl (float): Estimated amplitude.
    - bandwidth (float): Estimated bandwidth.
    - phi0 (float): Estimated phase at center frequency.
    - ampl0 (float): Estimated amplitude at center frequency.
    - delay (float): Estimated electrical delay.
    - amplitude_phase (float): Estimated average phase of the impedance.


    References:
    - https://github.com/UlysseREGLADE/abcd_rf_fit#3-estimation-of-the-electrical-delay
    """
    f0 = np.mean(x)
    ampl = max(np.abs(z)) - min(np.abs(z))
    bandwidth = np.sign(x[0]) * (x[-1] - x[0]) / 10
    phi0 = np.mean(np.angle(z))
    ampl0 = np.mean(np.abs(z))

    delay = 0
    amplitude_phase = np.mean(np.angle(z))

    return np.array([f0, ampl, bandwidth, phi0, ampl0, delay, amplitude_phase])


class LorentzComplex(FitLogic[LorentzParam]):  # type: ignore
    param: _t.Type[LorentzParam] = LorentzParam
    func = staticmethod(lorentz_func)
    _guess = staticmethod(lorentz_guess)  # type: ignore

    _test_ignore = True
    _doc_ignore = True
