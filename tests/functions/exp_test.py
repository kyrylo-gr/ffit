from typing import NamedTuple

import numpy as np

import ffit as ff

from .testing_protocol import FuncTestingProtocol


class Params(NamedTuple):
    amplitude: float
    rate: float
    offset: float


class ExpFrontTest(FuncTestingProtocol):
    @staticmethod
    def func(x, amplitude, rate, offset):
        return amplitude * np.exp(rate * x) + offset

    def test_basic_1234(self):
        params = Params(1, 2, 3)
        x, y = self.prepare_xy(params, x=np.linspace(0, 3, 100))
        res = ff.Exp.fit(x, y).res
        self.assert_almost_equal_tuple(params, res)

    def test_basic_1234_guess(self):
        params = Params(1, 2, 3)
        x, y = self.prepare_xy(params, x=np.linspace(0, 3, 100))
        res = ff.Exp.fit(x, y, guess=(1, 2, 3)).res
        self.assert_almost_equal_tuple(params, res)

    def test_different_range(self):
        for param in self.create_different_params(3, values=np.arange(-3, 3, 1)):
            for start in np.arange(-5, 5, 1):
                x, y = self.prepare_xy(Params(*param), x=np.linspace(start, start + 5, 100))
                res = ff.Exp.fit(x, y).res
                y_dif = max(y) - min(y)
                self.assert_almost_equal_tuple(
                    tuple(param),
                    res,
                    allow_error=[1e-3 / min(y_dif, 1), 1e-3 / min(y_dif, 1), None, None],
                    msg=f"start={start}",
                )
