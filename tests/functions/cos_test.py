from typing import NamedTuple

import numpy as np

import ffit as ff

from .testing_protocol import FuncTestingProtocol


def normalize_res(x):
    return [
        abs(x[0]),
        x[1],
        (x[2] + (np.pi if x[0] < 0 else 0)) % (2 * np.pi),
        x[3],
    ]


class Params(NamedTuple):
    amplitude: float
    frequency: float
    phi0: float
    offset: float


class CosFrontTest(FuncTestingProtocol):
    # func = cos_func
    @staticmethod
    def func(x, amplitude, frequency, phi0, offset):
        return amplitude * np.cos(2 * np.pi * x * frequency + phi0) + offset

    def test_basic_1234(self):
        params = Params(1, 2, 3, 4)
        x, y = self.prepare_xy(params)
        res = ff.Cos().fit(x, y).res
        self.assert_almost_equal_tuple(params, res)  # type: ignore

    def test_basic_1234_guess(self):
        params = Params(1, 2, 3, 4)
        x, y = self.prepare_xy(params)
        res = ff.Cos().fit(x, y, guess=(1, 2, 3, 3)).res
        self.assert_almost_equal_tuple(params, res)  # type: ignore

    def test_different_range(self):
        for param in self.create_different_params(4):
            param = normalize_res(param)
            freq = param[1]
            period = 1 / freq
            for t in np.arange(0.5, 5, 0.5):
                x = np.linspace(0, period * t, 100)
                x, y = self.prepare_xy(Params(*param), x=x)
                res = ff.Cos().fit(x, y).res
                self.assert_almost_equal_tuple(
                    tuple(param), res, max(param[0], param[-1]), msg=f"t={t}"  # type: ignore
                )

    # def test_different_range_noisy(self):
    #     for param in self.create_matrix(4, ranges=np.arange(-1, 1, 1)):
    #         param = normalize_res(param)
    #         freq = param[1]
    #         period = 1 / freq
    #         for t in np.arange(0.5, 5, 0.5):
    #             x = np.linspace(0, period * t, 100)
    #             x, y = self.prepare_xy(Params(*param), x=x)
    #             y += np.random.normal(0, 0.1 * param[0], len(y))
    #             res = ff.Cos().fit(x, y).res
    #             self.assert_almost_equal_tuple(
    #                 tuple(param),
    #                 res,
    #                 [max(param[0], param[-1]) * 12, 100, 100, max(param[0], param[-1]) * 12],
    #                 msg=f"t={t}",
    #             )
