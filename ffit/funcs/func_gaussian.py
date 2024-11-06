import typing as _t
from dataclasses import dataclass

import numpy as np

from ..fit_logic import FitLogic
from ..utils import _NDARRAY, ParamDataclass, check_min_len


@dataclass(frozen=True)
class GaussianParam(ParamDataclass):
    mu: float
    sigma: float
    amplitude: float
    offset: float
    std: "_t.Optional[GaussianParam]" = None


def gaussian_func(x, mu, sigma, amplitude, offset):
    return (
        amplitude
        * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        / np.sqrt(2 * np.pi)
        / sigma
        + offset
    )


def gaussian_guess(x, y, **kwargs):
    del kwargs
    if not check_min_len(x, y, 3):
        return np.ones(4)

    mu = np.mean(x)
    sigma = np.std(x) / 3
    amplitude = (np.max(y) - np.min(y)) * sigma * np.sqrt(2 * np.pi)
    offset = np.mean(y)

    return GaussianParam(
        mu=mu,
        sigma=sigma,  # type: ignore
        amplitude=amplitude,
        offset=offset,
    )


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array([x[0], np.abs(x[1]), x[2] * np.sign(x[1]), x[3]])


class Gaussian(FitLogic[GaussianParam]):  # type: ignore
    r"""Fit Hyperbola function.


    Function
    ---------

    $$
    a \cdot \frac{1}{\sqrt{2\pi}\sigma} \cdot \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) + b
    $$

        f(x) = (
            amplitude
            * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
            / np.sqrt(2 * np.pi)
            / sigma
            + offset
        )

    Example
    -------
        >>> import ffit as ff
        >>> res = ff.Gaussian().fit(x, y).res

        >>> res = ff.Gaussian().fit(x, y, guess=[1, 2, 3, 4]).plot(ax).res
        >>> mu = res.mu

    Final parameters
    -----------------
    - `mu`: float.
    - `sigma`: float.
    - `amplitude`: float.
    - `offset`: float.
    """

    param: _t.Type[GaussianParam] = GaussianParam

    func = staticmethod(gaussian_func)
    _guess = staticmethod(gaussian_guess)
    normalize_res = staticmethod(normalize_res_list)
