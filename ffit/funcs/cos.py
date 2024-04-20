import typing as _t

import numpy as np

from ..fit_logic import FitLogic


class CosParam(_t.NamedTuple):
    amplitude: float
    period: float
    offset: float


class Cos(FitLogic[CosParam]):
    param: _t.Type[CosParam] = CosParam

    @staticmethod
    def func(x, amplitude, period, offset):  # pylint: disable=W0221
        """Calculate the value of a cosine function at a given x-coordinate.

        Parameters:
        x (float): The x-coordinate at which to evaluate the function.
        amplitude (float): The amplitude of the cosine wave.
        period (float): The period of the cosine wave.
        offset (float): The vertical offset of the cosine wave.

        Returns:
        float: The value of the cosine function at the given x-coordinate.
        """
        return np.cos(2 * np.pi * x / period) * amplitude + offset

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
        off_guess = np.mean(y)
        amp_guess = np.abs(np.max(y - off_guess))
        nnn = 10 * len(y)
        fft_vals = np.fft.rfft(y - off_guess, n=nnn)
        fft_freqs = np.fft.rfftfreq(nnn, d=x[1] - x[0])

        freq_guess = np.abs(fft_freqs[np.argmax(np.abs(fft_vals))])
        sign_ = np.sign(np.real(fft_vals[np.argmax(np.abs(fft_vals))]))
        period = 1 / (freq_guess)

        return [sign_ * amp_guess, period, off_guess]
