import typing as _t

import jax.numpy as jnp

from ..fit_logic import FitLogic


class ComplexSpiralParam(_t.NamedTuple):
    amplitude0: float
    phi0: float
    freq: float
    tau: float
    offset: float


def complex_spiral_func(x, params: jnp.ndarray):
    """Complex spiral function.

    Parameters:
    - 0: amplitude0,
    - 1: phi0
    - 2: freq
    - 3: tau
    - 4: offset
    # TODO: Add complex offset phase
    """
    ampl = params[0] * jnp.exp(1j * params[1])
    return ampl * jnp.exp(1j * params[2] * 2 * jnp.pi * x - x / params[3]) + params[4]


class ComplexSpiral(FitLogic[ComplexSpiralParam]):
    param: _t.Type[ComplexSpiralParam] = ComplexSpiralParam
    func = complex_spiral_func

    @staticmethod
    def _guess(x, z, **kwargs):  # pylint: disable=W0237
        the_fft = jnp.fft.fft(z - z.mean())
        index_max = jnp.argmax(jnp.abs(the_fft))
        freq = jnp.fft.fftfreq(len(z), d=x[1] - x[0])[index_max]
        ampl = the_fft[index_max]

        return [
            (jnp.max(jnp.real(z)) - jnp.min(jnp.real(z))) / 2,
            jnp.angle(ampl),
            freq,
            jnp.max(x) / 2,
            (jnp.max(jnp.real(z)) + jnp.min(jnp.real(z))) / 2,
        ]
