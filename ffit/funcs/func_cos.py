import typing as _t

import jax
import jax.numpy as jnp
import numpy as np

from ..fit_logic import FitLogic


class CosParam(_t.NamedTuple):
    amplitude: float
    frequency: float
    phi0: float
    offset: float


@jax.jit
def cos_func_one(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Cosine function.
    Parameters:
    - 0: amplitude
    - 1: frequency
    - 2: phi0
    - 3: offset
    """
    return params[0] * jnp.cos(2 * jnp.pi * x * params[1] + params[2]) + params[3]


def normalize_res_list(x: _t.Sequence[float]) -> list:
    return [
        abs(x[0]),
        x[1],
        (x[2] + (np.pi if x[0] < 0 else 0)) % (2 * np.pi),
        x[3],
    ]


def cos_func(x, amplitude, frequency, phi0, offset):
    return amplitude * np.cos(2 * np.pi * x * frequency + phi0) + offset


@jax.jit
def cos_jac(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Jacobian of the cosine function."""
    return jnp.vstack(
        [
            jnp.cos(2 * jnp.pi * x * params[1] + params[2]),
            -2 * jnp.pi * x * jnp.sin(2 * jnp.pi * x * params[1] + params[2]) * params[0],
            -jnp.sin(2 * jnp.pi * x * params[1] + params[2]) * params[0],
            jnp.ones_like(x),
        ]
    )


# @jax.jit
def cos_guess(x, y, **kwargs):
    off_guess = np.mean(y)
    amp_guess = np.abs(np.max(y - off_guess))
    nnn = 10 * len(y)
    fft_vals = np.fft.rfft(y - off_guess, n=nnn)
    fft_freqs = np.fft.rfftfreq(nnn, d=x[1] - x[0])
    freq_max_index = np.argmax(np.abs(fft_vals))
    freq_guess = np.abs(fft_freqs[freq_max_index])
    sign_ = np.sign(np.real(fft_vals[freq_max_index]))
    phase = np.imag(fft_vals[freq_max_index])

    return np.array(normalize_res_list([sign_ * amp_guess, freq_guess, phase, off_guess]))


class Cos(FitLogic[CosParam]):
    param: _t.Type[CosParam] = CosParam
    func = cos_func
    normalize_res = normalize_res_list
    func_one = cos_func_one
    jac = cos_jac

    @staticmethod
    def _guess(x, y, **kwargs):
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
        return cos_guess(x, y, **kwargs)
