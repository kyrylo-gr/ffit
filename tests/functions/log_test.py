from typing import NamedTuple

import numpy as np

import ffit as ff

from .testing_protocol import FuncTestingProtocol


class Params(NamedTuple):
    amplitude: float
    rate: float
    offset: float


def normalize_res(x):
    return [x[0], max(abs(x[1]), 0.1), x[2]]


class LogFrontTest(FuncTestingProtocol):
    @staticmethod
    def func(x, amplitude, rate, offset):
        return amplitude * np.log(rate * x) + offset

    def test_basic_1234(self):
        params = Params(1, 2, 3)
        x, y = self.prepare_xy(params, x=np.linspace(1, 1000, 100))
        res = ff.Log().fit(x, y).res
        assert res is not None
        self.assert_almost_equal_tuple(params, tuple(res), allow_error=[None, 10, None])

    def test_basic_1234_guess(self):
        params = Params(1, 2, 3)
        x, y = self.prepare_xy(params, x=np.linspace(1, 3, 100))
        res = ff.Log().fit(x, y, guess=(1, 2, 3)).res
        assert res is not None
        self.assert_almost_equal_tuple(params, tuple(res))

    # def test_different_range(self):
    #     for param in self.create_different_params(3, values=np.arange(-3, 3, 1)):
    #         param = normalize_res(param)
    #         for start in np.arange(1, 5, 1):
    #             x, y = self.prepare_xy(Params(*param), x=np.linspace(start, start + 200, 100))
    #             res = ff.Log().fit(x, y).res
    #             self.assert_almost_equal_tuple(
    #                 tuple(param),
    #                 res,
    #                 allow_error=[None, 10, None],
    #                 msg=f"start={start}",
    #             )
