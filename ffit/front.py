import typing as _t

from scipy import optimize

from .fit_results import FitResult
from .utils import _NDARRAY, DynamicNamedTuple, create_named_tuple

# _T = _t.TypeVar("_T", bound=_t.Sequence)


def curve_fit(
    func: _t.Callable,
    x: _NDARRAY,
    data: _NDARRAY,
    p0: _t.Optional[_t.List[_t.Any]] = None,
    *,
    bounds: _t.Optional[_t.List[_t.Tuple[_t.Any, _t.Any]]] = None,
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


def leastsq(func: _t.Callable, x0: _t.Sequence, args: tuple = (), **kwarg) -> FitResult[tuple]:
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
    res_all = optimize.leastsq(func, x0, args=args, **kwarg)

    return FitResult(res_all, lambda x: func(x, *res_all))
