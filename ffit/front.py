import asyncio
import typing as _t

import numpy as np
from scipy import optimize

from .fit_results import FitResult
from .utils import _2DARRAY, _ARRAY, _NDARRAY, get_function_args_ordered


def _curve_fit(
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
    method: _t.Literal["leastsq", "curve_fit"] = "curve_fit",
    **kwargs,
) -> _NDARRAY:
    if method == "leastsq":

        def to_minimize(params):
            return np.abs(func(x, *params) - data)

        opt, cov, infodict, _, _ = optimize.leastsq(  # type: ignore
            to_minimize, p0, full_output=True, **kwargs
        )
        del cov, infodict
        return np.asarray(opt)

    elif method == "curve_fit":
        res_all = optimize.curve_fit(func, x, data, p0=p0, bounds=bounds, **kwargs)
        return np.asarray(res_all[0])

    raise ValueError("Invalid method argument. Use 'leastsq' or 'curve_fit'.")


async def _async_curve_fit(*args, **kwargs) -> _NDARRAY:
    return _curve_fit(*args, **kwargs)


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
    method: _t.Literal["leastsq", "curve_fit"] = "curve_fit",
    **kwargs,
) -> FitResult:
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

    res = _curve_fit(func, x, data, p0=p0, bounds=bounds, method=method, **kwargs)
    args_ordered = tuple(key for key, _ in get_function_args_ordered(func)[1:])
    return FitResult(
        np.asarray(res),
        lambda x: func(x, *res),
        x=x,
        data=data,
        keys=args_ordered,
    )


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
) -> FitResult:
    return curve_fit(func=func, x=x, data=data, p0=p0, bounds=bounds, **kwargs)


async def async_curve_fit_array(
    func: _t.Callable,
    x: _NDARRAY,
    data: _2DARRAY,
    p0: _t.Optional[_t.Sequence] = None,
    *,
    mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
    guess: _t.Optional[_ARRAY] = None,
    bounds: _t.Optional[
        _t.Union[_t.List[_t.Tuple[_t.Any, _t.Any]], _t.Tuple[_t.Any, _t.Any]]
    ] = (
        -np.inf,
        np.inf,
    ),
    axis: int = -1,
    method: _t.Literal["leastsq", "curve_fit"] = "curve_fit",
    **kwargs,
):

    # Convert x and data to numpy arrays
    x, data = np.asarray(x), np.asarray(data)
    if axis != -1:
        data = np.moveaxis(data, axis, -1)
    data_shape = data.shape
    selected_axis_len = data_shape[-1]
    data = data.reshape(-1, selected_axis_len)

    tasks = [
        _async_curve_fit(
            func=func,
            x=x,
            data=data[i],
            mask=mask,
            guess=guess,
            method=method,
            **kwargs,
        )
        for i in range(len(data))
    ]
    results = await asyncio.gather(*tasks)
    results = np.array(results)
    fit_param_len = results.shape[-1]
    results = results.reshape(data_shape[:-1] + (-1,))

    def fin_func(xx):
        return np.array(
            [func(xx, *res) for res in results.reshape(-1, fit_param_len)]
        ).reshape(data_shape[:-1] + (-1,))

    args_ordered = tuple(key for key, _ in get_function_args_ordered(func)[1:])

    return FitResult(results, fin_func, x=x, data=data, keys=args_ordered)


# def leastsq(func: _t.Callable, x0: _t.Sequence, args: tuple = (), **kwarg) -> FitResult:
#     """Perform a least squares optimization using the `leastsq` function from the `optimize` module.

#     This function returns [FitResult][ffit.fit_results.FitResult] see
#     the documentation for more information what is possible with it.

#     Args:
#         func: The objective function to minimize.
#         x0: The initial guess for the optimization.
#         args: Additional arguments to be passed to the objective function.
#         **kwarg: Additional keyword arguments to be passed to the `leastsq` function.

#     Returns:
#         A `FitResult` object containing the optimization result and a function to evaluate the optimized parameters.

#     """
#     res, cov = optimize.leastsq(func, x0, args=args, **kwarg)
#     # print(res)
#     return FitResult(
#         res,
#         cov=cov,  # type: ignore
#     )
