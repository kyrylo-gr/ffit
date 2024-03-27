from __future__ import annotations
from cmath import phase

from typing import NamedTuple

import numpy as np

from ..fit_utils import interpolate_on_regular_grid


class SinParams(NamedTuple):
    """Sin parameters"""
    amplitude: float = 1
    frequency: float = 1
    phase: float = 1
    offset: float = 1
    
    def __mul__(self, other: SinParams) -> SinParams:
        return SinParams(
                amplitude=self.amplitude*other.amplitude,
                frequency=self.frequency*other.frequency,
                phase=self.phase*other.phase,
                offset=self.offset*other.offset)
        
    def right_form(self) -> SinParams:
        amplitude, phase = (self.amplitude, self.phase) if self.amplitude>0 else (-self.amplitude, self.phase+np.pi)
        while(phase>np.pi):
            phase=phase-2*np.pi
        
        return SinParams(
                amplitude=amplitude,
                frequency=self.frequency,
                phase=phase,
                offset=self.offset)
    




class Sin:
    """Sin"""
    # @staticmethod
    @classmethod
    def guess(cls, x: np.ndarray, y: np.ndarray) -> SinParams:
        amplitude = np.max(y) - np.min(y)
        frequency = cls.guess_sin_freq(x, y)
        phase = 0
        offset = (np.max(y)+np.min(y))/2
    
        return SinParams(
            amplitude=amplitude,
            frequency=frequency,
            phase=phase,
            offset=offset
        )
        
    @staticmethod
    def func(x: np.ndarray, amplitude: float, frequency: float, phase: float, offset: float) -> np.ndarray:
        return amplitude*np.sin(frequency*x + phase) + offset
    
    @staticmethod
    def guess_sin_freq(x: np.ndarray, y: np.ndarray) -> float:
        
        x_interp, y_interp = interpolate_on_regular_grid(x, y)
        
        fft = np.fft.fft(y_interp)
        freqs = np.fft.fftfreq(len(x_interp), d=x_interp[1]-x_interp[0])
        return 2*np.pi*freqs[1+np.argmax(fft[1:len(fft)//2])] # cuttoff zero element (i.e. the offset) and take only positive freqs
