from typing import Any, Callable

import numpy as np


def chisq(func: Callable, x: np.ndarray, data: np.ndarray, params: Any) -> np.ndarray:
    return np.sum((func(x, *params) - data) ** 2)


def interpolate_on_regular_grid(x, y):
    x_inter = np.arange(np.min(x), np.max(x), np.min(np.diff(x)))
    return x_inter, np.interp(x_inter, x, y)
