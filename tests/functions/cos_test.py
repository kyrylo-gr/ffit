import unittest
from typing import NamedTuple

import numpy as np

import ffit as ff


def cos_func(x, amplitude, frequency, phi0, offset):
    return amplitude * np.cos(2 * np.pi * x * frequency + phi0) + offset


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


class CosFrontTest(unittest.TestCase):

    def prepare_xy(
        self, params: Params, x: np.ndarray | None | int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if x is None:
            x = 100
        if isinstance(x, int):
            x = np.linspace(0, x, 10 * x)
        y = cos_func(x, params.amplitude, params.frequency, params.phi0, params.offset)
        return x, y

    def assert_almost_equal_tuple(
        self,
        t1: tuple | None,
        t2: tuple | None,
        normalization_constant: list[float] | float | None = None,
        msg: str = "",
    ):
        assert t1 is not None
        assert t2 is not None
        self.assertEqual(len(t1), len(t2))

        for i, (v1, v2) in enumerate(zip(t1, t2)):
            if normalization_constant is None:
                norm = (v1 + v2) / 2
            elif isinstance(normalization_constant, (list, tuple)):
                norm = normalization_constant[i]
            else:
                norm = normalization_constant

            error = (v1 - v2) / norm
            self.assertAlmostEqual(error, 0, places=2, msg=f"{v1}<>{v2} => {t1} != {t2}; {msg}")

    def test_basic_1234(self):
        params = Params(1, 2, 3, 4)
        x, y = self.prepare_xy(params)
        res = ff.Cos.fit(x, y).res
        self.assert_almost_equal_tuple(params, res)

    def test_basic_1234_guess(self):
        params = Params(1, 2, 3, 4)
        x, y = self.prepare_xy(params)
        res = ff.Cos.fit(x, y, guess=(1, 2, 3, 3)).res
        self.assert_almost_equal_tuple(params, res)

    def create_matrix(self, params_len: int, ranges=None):
        if ranges is None:
            ranges = np.arange(-5, 9, 1)
        param = np.ones(params_len)
        for i in range(params_len):
            for r in ranges:
                param[i] = 10.0 ** (r)  # (1 + 9 * np.random.rand()) *
                yield param

    def test_different_range(self):
        for param in self.create_matrix(4):
            param = normalize_res(param)
            freq = param[1]
            period = 1 / freq
            for t in np.arange(0.5, 5, 0.5):
                x = np.linspace(0, period * t, 100)
                x, y = self.prepare_xy(Params(*param), x=x)
                res = ff.Cos.fit(x, y).res
                self.assert_almost_equal_tuple(
                    tuple(param), res, max(param[0], param[-1]), msg=f"t={t}"
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
    #             res = ff.Cos.fit(x, y).res
    #             self.assert_almost_equal_tuple(
    #                 tuple(param),
    #                 res,
    #                 [max(param[0], param[-1]) * 12, 100, 100, max(param[0], param[-1]) * 12],
    #                 msg=f"t={t}",
    #             )


if __name__ == "__main__":
    unittest.main()
