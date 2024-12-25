import asyncio
import unittest

import numpy as np

import ffit as ff


class CosFrontTest(unittest.TestCase):
    def test_async_array_fit_2D(self):
        x = np.linspace(-1, 1, 100)
        y = np.tile(np.sin(2 * np.pi * x), (5, 1))

        fit = asyncio.run(ff.Cos().async_array_fit(x, y, axis=-1))

        self.assertTrue(np.isclose(y, fit.res_func(x)).all())

    def test_async_array_fit_3D(self):
        x = np.linspace(-1, 1, 100)
        y = np.tile(np.sin(2 * np.pi * x), (5, 7, 1))

        fit = asyncio.run(ff.Cos().async_array_fit(x, y, axis=-1))

        self.assertTrue(np.isclose(y, fit.res_func(x)).all())
