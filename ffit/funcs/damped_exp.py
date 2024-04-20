import typing as _t

import numpy as np

from ..fit_logic import FitLogic


class DampedExpParam(_t.NamedTuple):
    amplitude: float
    tau: float
    offset: float


class DampedExp(FitLogic[DampedExpParam]):
    param: _t.Type[DampedExpParam] = DampedExpParam

    @staticmethod
    def func(x, amplitude, tau, offset):
        return np.exp(-x / tau) * amplitude + offset

    @staticmethod
    def _guess(x, y, **kwargs):
        return np.max(y) - np.min(y), (np.max(x) - np.min(x)) / 5, np.mean(y)
