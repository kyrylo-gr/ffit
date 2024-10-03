import asyncio
import typing as _t

import numpy as np
from scipy import optimize

from .fit_results import FitArrayResult, FitResult
from .utils import _2DARRAY, _NDARRAY, DynamicNamedTuple, create_named_tuple

# _T = _t.TypeVar("_T", bound=_t.Sequence)


def curve_fit(
    func: _t.Callable,
    x: _NDARRAY,
    data: _NDARRAY,
    p0: _t.Optional[_t.List[_t.Any]] = None,
    *,
    bounds: _t.Optional[
        _t.Union[_t.List[_t.Tuple[_t.Any, _t.Any]], _t.Tuple[_t.Any, _t.Any]]
    ] = (
        -np.inf,
        np.inf,
    ),
    **kwargs,
) -> FitResult[DynamicNamedTuple]:
    """Fit a curve with curve_fit method.

    This function returns [FitResult][ffit.fit_results.FitResult] see
    the documentation for more information what is possible with it.

    Args:
        fit_func: Function to fit.
        x: x data.
        data: data to fit.
        p0: Initial guess for the parameters.
        bounds: Bounds for the parameters.
        **kwargs: Additional keyword arguments to curve_fit.

    Returns:
        FitResult: Fit result.
    """
    res_all = optimize.curve_fit(func, x, data, p0=p0, bounds=bounds, **kwargs)
    res = create_named_tuple(func, res_all[0])
    return FitResult(res, lambda x: func(x, *res), x=x, data=data)


async def async_curve_fit(
    func: _t.Callable,
    x: _NDARRAY,
    data: _NDARRAY,
    p0: _t.Optional[_t.List[_t.Any]] = None,
    *,
    bounds: _t.Optional[
        _t.Union[_t.List[_t.Tuple[_t.Any, _t.Any]], _t.Tuple[_t.Any, _t.Any]]
    ] = (
        -np.inf,
        np.inf,
    ),
    **kwargs,
) -> FitResult[DynamicNamedTuple]:
    return curve_fit(func=func, x=x, data=data, p0=p0, bounds=bounds, **kwargs)


async def async_curve_fit_array(
    func: _t.Callable,
    x: _NDARRAY,
    data: _2DARRAY,
    p0: _t.Optional[_t.List[_t.Any]] = None,
    *,
    bounds: _t.Optional[
        _t.Union[_t.List[_t.Tuple[_t.Any, _t.Any]], _t.Tuple[_t.Any, _t.Any]]
    ] = (
        -np.inf,
        np.inf,
    ),
    **kwargs,
):
    tasks = [
        async_curve_fit(func=func, x=x, data=data[i], p0=p0, bounds=bounds, **kwargs)
        for i in range(len(data))
    ]
    results = await asyncio.gather(*tasks)

    def res_func(y):
        return np.array([res.res_func(y) for res in results])

    return FitArrayResult(results, res_func)


def leastsq(
    func: _t.Callable, x0: _t.Sequence, args: tuple = (), **kwarg
) -> FitResult[tuple]:
    """Perform a least squares optimization using the `leastsq` function from the `optimize` module.

    This function returns [FitResult][ffit.fit_results.FitResult] see
    the documentation for more information what is possible with it.

    Args:
        func: The objective function to minimize.
        x0: The initial guess for the optimization.
        args: Additional arguments to be passed to the objective function.
        **kwarg: Additional keyword arguments to be passed to the `leastsq` function.

    Returns:
        A `FitResult` object containing the optimization result and a function to evaluate the optimized parameters.
    """

    res, cov = optimize.leastsq(func, x0, args=args, **kwarg)
    # print(res)
    return FitResult(res, lambda x: func(res), cov=cov)
