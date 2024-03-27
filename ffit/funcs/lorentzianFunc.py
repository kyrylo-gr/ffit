from __future__ import annotations
from typing import NamedTuple
import numpy as np


class LorentzianParams(NamedTuple):
    """Sin parameters"""
    amplitude: float = 1
    bw: float = 1
    x0: float = 1
    offset: float = 1
    
    def __mul__(self, other: LorentzianParams) -> LorentzianParams:
        return LorentzianParams(
                    amplitude=self.amplitude*other.amplitude,
                    bw=self.bw*other.bw,
                    x0=self.x0*other.x0,
                    offset=self.offset*other.offset)


class Lorentzian:
    """Sin"""
    @staticmethod
    def guess(x: np.ndarray, y: np.ndarray) -> LorentzianParams:
        amplitude = np.max(y) - np.min(y)
        bw = Lorentzian().guess_lorentzian_bw(x, y)
        x0 = Lorentzian().guess_lorentzian_x0(x, y)
        offset = (np.max(y)+np.min(y))/2
    
        return LorentzianParams(
            amplitude=amplitude,
            bw=bw,
            x0=x0,
            offset=offset
        )
        
    @staticmethod
    def func(x: np.ndarray, amplitude, bw, x0, offset) -> np.ndarray:
        return amplitude*1/np.pi*1/2*bw/((x-x0)**2+(bw/2)**2)+offset
    
    @staticmethod 
    def guess_lorentzian_bw(x: np.ndarray, _) -> float:
        return x[-1] - x[0]

    @staticmethod
    def guess_lorentzian_x0(x: np.ndarray, y: np.ndarray) -> float:
        return x[np.argmax(y)] 