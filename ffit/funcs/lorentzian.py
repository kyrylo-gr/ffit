import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _NDARRAY, ParamDataclass, check_min_len


@dataclass(frozen=True)
class LorentzianParam(ParamDataclass):
    """Lorentzian parameters.

    Attributes:
        amplitude (float):
            The height of the peak.
        gamma (float):
            The half-width at half-maximum.
        x0 (float):
            The position of the peak.
        offset (float):
            The baseline offset.

    Additional attributes:
        sigma (float):
            The full width at half-maximum.
    """

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
    lowest_amp = np.mean(sorted_data[:average_size, 1])
    amplitude_diff = np.mean(sorted_data[-average_size:, 1]) - lowest_amp
    gamma = np.std(sorted_data[:average_size, 0])
    direction = (
        1
        if np.std(sorted_data[: len(sorted_data) // 2, 0])
        > np.std(sorted_data[-len(sorted_data) // 2 :, 0])
        else -1
    )

    x0 = (
        np.mean(sorted_data[-average_size:, 0])
        if direction == 1
        else np.mean(sorted_data[:average_size, 0])
    )

    return np.array([direction * amplitude_diff, gamma, x0, lowest_amp])


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array([x[0], abs(x[1]), x[2], x[3]])


class Lorentzian(FitLogic[LorentzianParam]):  # type: ignore
    r"""Lorentzian function.
    ---------

    $$
    f(x) = A * \frac{\gamma^2}{(x-x_0)^2 + \gamma^2} + A_0
    $$

        f(x) = amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset

    In this notation, the width at half-height: $\sigma = 2\gamma$


    Final parameters
    -----------------
    The final parameters are given by [`LorentzianParam`](../lorentzian_param/) dataclass.


    """

    param: _t.Type[LorentzianParam] = LorentzianParam
    func = staticmethod(lorentzian_func)
    _guess = staticmethod(lorentzian_guess)
    normalize_res = staticmethod(normalize_res_list)

    _example_param = (5, 1, 3, 2)
    _example_x_min = 0
    _example_x_max = 6
