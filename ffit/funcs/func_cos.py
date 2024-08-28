import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _ARRAY, _NDARRAY, ParamDataclass, check_min_len


@dataclass(frozen=True)
class CosParam(ParamDataclass):
    amplitude: float
    frequency: float
    phi0: float
    offset: float
    std: "_t.Optional[CosParam]" = None


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array(
        [
            abs(x[0]),
            x[1],
            (x[2] + (np.pi if x[0] < 0 else 0)) % (2 * np.pi),
            x[3],
        ]
    )


def cos_func(
    x: _NDARRAY, amplitude: float, frequency: float, phi0: float, offset: float
):
    return amplitude * np.cos(2 * np.pi * x * frequency + phi0) + offset


def std_monte_carlo(
    x: _NDARRAY,
    func: _t.Callable,
    means: _ARRAY,
    stds: _ARRAY,
    n_simulations: int = 10_000,
) -> _NDARRAY:
    # Arrays to hold the results of each simulation
    simulated_functions = np.zeros((n_simulations, len(x)))
    # Sampling from normal distribution
    values = np.array(
        [np.random.normal(m, s, n_simulations) for m, s in zip(means, stds)]
    )
    # Monte Carlo simulation
    for i in range(n_simulations):
        simulated_functions[i, :] = func(x, *values[:, i])

    return np.std(simulated_functions, axis=0)


def cos_error(
    x: _NDARRAY,
    amplitude: float,
    frequency: float,
    phi0: float,
    offset: float,
    amplitude_std: float,
    frequency_std: float,
    phi0_std: float,
    offset_std: float,
):
    # del offset
    # amplitude_error = np.cos(2 * np.pi * x * frequency + phi0) * amplitude_std
    # frequency_error = (
    #     amplitude * np.sin(2 * np.pi * x * frequency + phi0) * 2 * np.pi * x
    # ) * frequency_std
    # phi0_error = amplitude * np.sin(2 * np.pi * x * frequency + phi0) * phi0_std
    # offset_error = offset_std
    # return np.sqrt(
    #     amplitude_error**2 + frequency_error**2 + phi0_error**2 + offset_error**2
    # ) # Mean values of the parameters
    return std_monte_carlo(
        x,
        cos_func,
        [amplitude, frequency, phi0, offset],
        [amplitude_std, frequency_std, phi0_std, offset_std],
    )
    # # Number of Monte Carlo simulations
    # n_simulations = 10000

    # # Arrays to hold the results of each simulation
    # simulated_functions = np.zeros((n_simulations, len(x)))

    # # Sampling from normal distribution
    # amplitudes = np.random.normal(amplitude, amplitude_std, n_simulations)
    # frequencies = np.random.normal(frequency, frequency_std, n_simulations)
    # phi0s = np.random.normal(phi0, phi0_std, n_simulations)
    # offsets = np.random.normal(offset, offset_std, n_simulations)

    # # Monte Carlo simulation
    # for i in range(n_simulations):
    #     simulated_functions[i, :] = cos_func(
    #         x, amplitudes[i], frequencies[i], phi0s[i], offsets[i]
    #     )

    # return np.std(simulated_functions, axis=0)


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


class Cos(FitLogic[CosParam]):  # type: ignore
    r"""Fit Cos function.


    Function
    ---------

    $$
    f(x) = A * cos(2 * pi * \omega* x + \phi_0) + A_0
    $$

        f(x) = amplitude * cos(2 * pi * frequency * x + phi0) + offset

    Example
    ---------
        >>> import ffit as ff
        >>> res = ff.Cos.fit(x, y).res

        >>> res = ff.Cos.fit(x, y, guess=[1, 2, 3, 4]).plot(ax).res
        >>> amplitude = res.amplitude

    Final parameters
    -----------------
    - `amplitude`: float.
        The amplitude.
    - `frequency`: float.
        The frequency in 1/[x] units.
    - `phi0`: float.
        The phase inside cos.
    - `offset`: float.
        The global offset.

    """

    param: _t.Type[CosParam] = CosParam
    func = staticmethod(cos_func)
    func_std = staticmethod(cos_error)

    normalize_res = staticmethod(normalize_res_list)
    _guess = staticmethod(cos_guess)
