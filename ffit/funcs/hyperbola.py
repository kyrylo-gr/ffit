import typing as _t

import numpy as np

from ..fit_logic import FitLogic


class HyperbolaParam(_t.NamedTuple):
    semix: float
    semiy: float
    x0: float
    y0: float


class Hyperbola(FitLogic[HyperbolaParam]):
    param: _t.Type[HyperbolaParam] = HyperbolaParam
    _offset: bool = False

    @staticmethod
    def func(x, semix, semiy, x0, y0):
        return y0 + np.sign(semiy) * np.sqrt(semiy**2 + (semiy**2 / semix**2) * (x - x0) ** 2)

    @staticmethod
    def _guess(x, y, **kwargs):
        if len(y) == 0 or len(x) == 0:
            return HyperbolaParam(
                semix=0,
                semiy=1,
                x0=0,
                y0=0,
            )
        direction = kwargs.get("direction")
        if direction is None:
            cumsum_len = len(y) // 10
            if cumsum_len == 0:
                cumsum_len = 1

            smoth_y = np.convolve(y, np.ones(cumsum_len) / cumsum_len, mode="valid")
            smoth_y = np.diff(smoth_y)
            direction = 1 if np.mean(smoth_y[:cumsum_len]) > np.mean(smoth_y[-cumsum_len:]) else -1
        # print("direction", direction)
        x0 = x[np.argmax(y)] if direction > 0 else x[np.argmin(y)]
        y0 = np.max(y) if direction > 0 else np.min(y)
        return HyperbolaParam(
            semix=np.std(x),  # type: ignore
            semiy=-np.std(y) * direction,  # type: ignore
            x0=x0,
            y0=y0,  # np.mean(y),
        )
