from typing import NamedTuple

import numpy as np

import ffit as ff

from .testing_protocol import FuncTestingProtocol


class Params(NamedTuple):
    semix: float
    semiy: float
    x0: float
    y0: float


class HyperbolaFrontTest(FuncTestingProtocol):
    @staticmethod
    def func(x, semix, semiy, x0, y0):
        return y0 + semiy * np.sqrt(1 + ((x - x0) / semix) ** 2)

    def test_basic_1234(self):
        params = Params(1, 2, 3, 4)
        x, y = self.prepare_xy(params)
        res = ff.Hyperbola.fit(x, y).res
        self.assert_almost_equal_tuple(params, res)

    def test_basic_1234_guess(self):
        params = Params(1, 2, 3, 4)
        x, y = self.prepare_xy(params)
        res = ff.Hyperbola.fit(x, y, guess=(1, 2, 3, 3)).res
        self.assert_almost_equal_tuple(params, res)

    def test_different_range(self):
        for param in self.create_different_params(4, ranges=np.arange(-3, 3, 1)):
            for t in np.arange(0.1, 5, 0.2):
                lims = (param[2] - param[0] * t), (param[2] + param[0] * t)
                x = np.linspace(*lims, 100)
                x, y = self.prepare_xy(Params(*param), x=x)
                res = ff.Hyperbola.fit(x, y).res
                y_dif = max(y) - min(y)
                self.assert_almost_equal_tuple(
                    tuple(param),
                    res,
                    allow_error=[1e-3 / min(y_dif, 1), 1e-3 / min(y_dif, 1), None, None],
                    msg=f"t={t}",
                )
