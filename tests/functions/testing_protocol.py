import unittest
from typing import Callable

import numpy as np


class FuncTestingProtocol(unittest.TestCase):
    func: Callable
    normalize: Callable
    error: float = 0.05

    def prepare_xy(
        self, params: tuple, x: np.ndarray | None | int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if x is None:
            x = 100
        if isinstance(x, int):
            x = np.linspace(0, x, 10 * x)
        y = self.func(x, *params)
        return x, y

    def assert_almost_equal_tuple(
        self,
        t1: tuple | None,
        t2: tuple | None,
        normalization_constant: list[float] | float | None = None,
        allow_error: list[float | None] | float | None = None,
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

            if norm > self.error:
                error_ = (v1 - v2) / norm
            else:
                error_ = v1 - v2

            if allow_error is None:
                error = self.error
            elif isinstance(allow_error, (list, tuple)):
                error = allow_error[i]
                if error is None:
                    error = self.error
            else:
                error = allow_error

            self.assertTrue(
                error_ < error,
                msg=f"({v1}-{v2})/{norm} = {error_} > {error} => {t1} != {t2}; {msg}",
            )

    def create_different_params(self, params_len: int, ranges=None, values=None):
        if values is None:
            if ranges is None:
                ranges = np.arange(-5, 9, 1)
            values = 10.0**ranges
        param = np.ones(params_len)
        for i in range(params_len):
            for v in values:
                param[i] = v  # (1 + 9 * np.random.rand()) *
                yield param
