import typing as _t

import jax
import jax.numpy as jnp
import numpy as np

from ..fit_logic import FitLogic


class LineParam(_t.NamedTuple):
    offset: float
    amplitude: float


@jax.jit
def line_func(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    return params[0] + params[1] * x


@jax.jit
def line_jac(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    del params
    return jnp.array([jnp.ones_like(x), x])


class Line(FitLogic[LineParam]):
    param: _t.Type[LineParam] = LineParam
    func = line_func
    jac = line_jac

    @staticmethod
    def _guess(x, y, **kwargs):
        average_size = max(len(y) // 10, 1)
        y1 = np.average(y[:average_size])
        y2 = np.average(y[-average_size:])

        amplitude = (y2 - y1) / (x[-1] - x[0])
        offset = y1 - x[0] * amplitude

        return np.array([offset, amplitude])
