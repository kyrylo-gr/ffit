import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import _NDARRAY, FuncParamClass, check_min_len, convert_param_class

__all__ = ["Gaussian"]

_T = _t.TypeVar("_T")


class GaussianParam(_t.Generic[_T], FuncParamClass):
    """Gaussian parameters.

    Attributes:
    -----------------
        mu (float):
            The  Gaussian peak.
        sigma (float):
            The sqrt of the standard deviation.
        amplitude (float):
            Teh normalization factor.
        offset (float):
            The baseline offset from zero.

    Attention to use `sigma` and not `std`, as it is reserved for standard deviation of the parameters.
    """

    keys = ("mu", "sigma", "amplitude", "offset")

    mu: _T
    sigma: _T
    amplitude: _T
    offset: _T


class GaussianResult(GaussianParam, FitResult[GaussianParam]):
    param_class = convert_param_class(GaussianParam)


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

    sorted_indices = np.argsort(y)[::-1]
    mu = np.array(x)[sorted_indices][: min(max(len(x) // 10, 100), 1)].mean()

    # mu = np.mean(x)
    sigma = np.std(x) / 3
    amplitude = (np.max(y) - np.min(y)) * sigma * np.sqrt(2 * np.pi)
    offset = np.mean(y)

    return np.array([mu, sigma, amplitude, offset])


def normalize_res_list(x: _t.Sequence[float]) -> _NDARRAY:
    return np.array([x[0], np.abs(x[1]), x[2] * np.sign(x[1]), x[3]])


class Gaussian(FitLogic[GaussianResult]):  # type: ignore
    r"""Gaussian function.
    ---------

    $$
    a \cdot \frac{A}{\sqrt{2\pi}\sigma} \cdot \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) + b
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
    The final parameters are given by [`GaussianParam`](../gaussian_param/) dataclass.
    """

    _result_class: _t.Type[GaussianResult] = GaussianResult

    func = staticmethod(gaussian_func)
    _guess = staticmethod(gaussian_guess)
    normalize_res = staticmethod(normalize_res_list)

    _example_param = (0.2, 0.2, 0.2, 0.2)

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        mu: float = None,  # type: ignore
        sigma: float = None,  # type: ignore
        amplitude: float = None,  # type: ignore
        offset: float = None,  # type: ignore
    ) -> "Gaussian": ...

    @classmethod
    def mask(cls, **kwargs) -> "Gaussian":
        return super().mask(**kwargs)
