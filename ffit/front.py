import asyncio
import typing as _t

import numpy as np
from scipy import optimize

from .fit_results import FitResult
from .utils import (
    _2DARRAY,
    _ARRAY,
    _NDARRAY,
    get_function_args_ordered,
    get_random_array_permutations,
    std_monte_carlo,
)


def _curve_fit(
    func: _t.Callable,
    x: _NDARRAY,
    data: _NDARRAY,
    p0: _t.Optional[_ARRAY] = None,
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
        if bounds is not None and bounds != (-np.inf, np.inf):
            raise ValueError("bounds are not supported for leastsq method")

        def to_minimize(params):
            return np.abs(func(x, *params) - data).flatten()

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
    res = np.asarray(res)

    # Get ordered parameter names
    args_default = get_function_args_ordered(func)[1:]
    args_ordered = tuple(key for key, _ in args_default)
    values_ordered = tuple(val for _, val in args_default)

    if len(res) != len(values_ordered):
        res = np.concatenate([res, values_ordered[len(res) :]])

    return FitResult(
        res,
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

    # Get ordered parameter names
    args_default = get_function_args_ordered(func)[1:]
    args_ordered = tuple(key for key, _ in args_default)
    values_ordered = tuple(val for _, val in args_default)

    if len(results) != len(values_ordered):
        results = np.concatenate([results, values_ordered[len(results) :]])

    return FitResult(results, fin_func, x=x, data=data, keys=args_ordered)


def bootstrap_curve_fit(
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
    num_of_permutations: _t.Optional[int] = None,
    **kwargs,
) -> FitResult:
    """Fit a curve with curve_fit method using bootstrapping for error estimation.

    This function returns [FitResult][ffit.fit_results.FitResult] see
    the documentation for more information what is possible with it.

    Args:
        func: Function to fit.
        x: x data.
        data: data to fit.
        p0: Initial guess for the parameters.
        bounds: Bounds for the parameters.
        method: The fitting method to use (default: "curve_fit").
        num_of_permutations: Number of bootstrap iterations.
        **kwargs: Additional keyword arguments to curve_fit.

    Returns:
        FitResult: The result of the fit, including the fitted parameters and their uncertainties.
    """
    # Convert x and data to numpy arrays
    x, data = np.asarray(x), np.asarray(data)

    # Get initial fit to use as starting point for bootstrap iterations
    initial_res = _curve_fit(
        func, x, data, p0=p0, bounds=bounds, method=method, **kwargs
    )

    # Determine number of permutations if not specified
    total_elements = len(x)
    if num_of_permutations is None:
        num_of_permutations = int(min(max(total_elements / 10, 1_000), 5_000))

    # Run bootstrap iterations
    all_res = []
    for xx, yy in get_random_array_permutations(x, data, num_of_permutations):
        res = _curve_fit(
            func,
            xx,
            yy,
            p0=initial_res,
            bounds=bounds,
            method=method,
            **kwargs,
        )
        all_res.append(res)

    # Calculate mean and standard deviation of bootstrap results
    res_means = np.mean(all_res, axis=0)
    bootstrap_std = np.std(all_res, axis=0)

    # Get ordered parameter names
    args_default = get_function_args_ordered(func)[1:]
    args_ordered = tuple(key for key, _ in args_default)
    values_ordered = tuple(val for _, val in args_default)

    if len(res_means) != len(values_ordered):
        res_means = np.concatenate([res_means, values_ordered[len(res_means) :]])
        bootstrap_std = np.concatenate(
            [bootstrap_std, np.zeros_like(values_ordered[len(bootstrap_std) :])]
        )

    # Return FitResult with bootstrap statistics
    return FitResult(
        res_means,
        lambda x: func(x, *res_means),
        x=x,
        data=data,
        keys=args_ordered,
        stderr=bootstrap_std,
        stdfunc=lambda x: std_monte_carlo(x, func, res_means, bootstrap_std),
    )


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
