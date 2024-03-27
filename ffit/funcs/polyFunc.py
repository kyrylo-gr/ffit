from __future__ import annotations
from typing import NamedTuple
import numpy as np


class PolynomialParams(NamedTuple):
    """Sin parameters"""
    amplitudes: list[float] = []


class Polynomial:
    """Polynomial"""

    def __init__(self, power=1):
        self.power = power

    def guess(self, x: np.ndarray, y: np.ndarray) -> PolynomialParams:
        amplitudes = [1.]*(self.power+1)

        return PolynomialParams(
            amplitudes=amplitudes
        )

    @staticmethod
    def func(x: np.ndarray, *amplitudes: float) -> np.ndarray:
        # if len(amplitudes)==1:
        #     amplitudes=amplitudes[0]
        return sum(amp*(x**(i)) for i, amp in enumerate(amplitudes))
    # type: ignore
