from typing import NamedTuple

import numpy as np

import ffit as ff

from .testing_protocol import FuncTestingProtocol


class Params(NamedTuple):
    amplitude: float
    gamma: float
    x0: float
    offset: float


class LorentzianFrontTest(FuncTestingProtocol):
    @staticmethod
    def func(x, amplitude: float, gamma: float, x0: float, offset: float):
        return amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset

    def test_basic_1234(self):
        params = Params(1, 2, 3, 4)
        x, y = self.prepare_xy(params)
        res = ff.Lorentzian.fit(x, y).res
        self.assert_almost_equal_tuple(params, res)

    def test_basic_1234_guess(self):
        params = Params(1, 2, 3, 4)
        x, y = self.prepare_xy(params)
        res = ff.Lorentzian.fit(x, y, guess=(1, 2, 3, 3)).res
        self.assert_almost_equal_tuple(params, res)

    def test_different_range(self):
        for param in self.create_different_params(4, ranges=np.arange(-3, 3, 1)):
            for t in np.arange(0.1, 5, 0.2):
                lims = (param[2] - param[1] * t), (param[2] + param[1] * t)

                x = np.linspace(*lims, int(max(sum(lims) / param[1], 100)))

                # x = np.arange(param[-1] * (1 - t), param[-1] * (1 + t), param[1] * t)
                x, y = self.prepare_xy(Params(*param), x=x)
                res = ff.Lorentzian.fit(x, y).res
                self.assert_almost_equal_tuple(
                    tuple(param),
                    res,
                    [
                        max(param[0], param[-1]),
                        max(1 / ((lims[1] - lims[0]) / param[1]), 1),
                        1,  # x[-1] - x[0],
                        max(param[0], param[-1]),
                    ],
                    msg=f"t={t}",
                )
