import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _NDARRAY, ParamDataclass, check_min_len


@dataclass(frozen=True)
class LorentzianParam(ParamDataclass):
    amplitude: float
    gamma: float
    x0: float
    offset: float

    std: "_t.Optional[LorentzianParam]" = None

    @property
    def sigma(self):
        return self.gamma * 2


def lorentzian_func(
    x: _NDARRAY, amplitude: float, gamma: float, x0: float, offset: float
):
    return amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset


def lorentzian_guess(x: _NDARRAY, y: _NDARRAY, **kwargs):
    del kwargs
    if not check_min_len(x, y, 3):
        return np.zeros(4)

    average_size = max(len(y) // 10, 1)

    data = np.array([x, y]).T
    sorted_data = data[data[:, 1].argsort()]
    offset = np.mean(sorted_data[:average_size, 1])
    amplitude = np.mean(sorted_data[-average_size:, 1]) - offset
    gamma = np.std(sorted_data[:average_size, 0])
    x0 = np.mean(sorted_data[-average_size:, 0])
    return np.array([amplitude, gamma, x0, offset])


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array([x[0], abs(x[1]), x[2], x[3]])


class Lorentzian(FitLogic[LorentzianParam]):  # type: ignore
    r"""Fit Lorentzian function.


    Function
    ---------

    $$
    f(x) = A * \frac{\gamma^2}{(x-x_0)^2 + \gamma^2} + A_0
    $$

        f(x) = amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset

    In this notation, the width at half-height: $\sigma = 2\gamma$

    Example
    ---------
        >>> import ffit as ff
        >>> res = ff.Lorentzian.fit(x, y).res

        >>> res = ff.Lorentzian.fit(x, y, guess=[1, 2, 3, 4]).plot(ax).res
        >>> amplitude = res.amplitude

    Final parameters
    -----------------
    - `amplitude`: float.
        The height of the max.
    - `gamma`: float.
        The half-width at half-maximum.
    - `x0`: float.
        The position of the maximum.
    - `offset`: float.
        The global offset.
    - `sigma`: float.
        The full width at half-maximum.

    """

    param: _t.Type[LorentzianParam] = LorentzianParam
    func = staticmethod(lorentzian_func)
    _guess = staticmethod(lorentzian_guess)
    normalize_res = staticmethod(normalize_res_list)
