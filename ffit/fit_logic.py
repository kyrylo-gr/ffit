import abc

# import numpy as np
import asyncio
import typing as _t

# import jax.numpy as jnp
import numpy as np
from scipy import optimize

from .fit_results import FitArrayResult, FitResult
from .utils import _ARRAY, _NDARRAY, get_mask, param_len

_T = _t.TypeVar("_T", bound=_t.Sequence)


class FitLogic(_t.Generic[_T]):
    """
    A generic class for fitting logic.

    Parameters:
    - param: The parameter type for the fit.

    Methods:
    - __init__: Initializes the FitLogic instance.
    - func: Abstract method for the fitting function.
    - _guess: Abstract method for guessing initial fit parameters.
    - fit: Fits the data using the specified fitting function.
    - sfit: Fits the data using the specified fitting function with simulated annealing.
    - guess: Guesses the initial fit parameters.
    - error: Calculates the error between the fitted function and the data.
    - get_mask: Returns a mask array based on the provided mask or threshold.

    Attributes:
    - param: The parameter type for the fit.
    """

    param: abc.ABCMeta

    def __init__(self, *args, **kwargs):
        """Initialize the FitLogic instance.

        Parameters:
        - args: Positional arguments.
        - kwargs: Keyword arguments.
        """
        del args
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)

    func: _t.Callable[..., _NDARRAY]
    normalize_res: _t.Optional[_t.Callable[[_t.Sequence[float]], _t.Sequence[float]]]
    jac: _t.Optional[_t.Callable[..., _NDARRAY]] = None

    # @abc.abstractmethod
    # @staticmethod
    # def func(x, *args, **kwargs):
    #     """Abstract method for the fitting function.

    #     Parameters:
    #     - x: The independent variable.
    #     - args: Positional arguments.
    #     - kwargs: Keyword arguments.
    #     """

    @staticmethod
    def _guess(x, y, **kwargs):
        """Abstract method for guessing initial fit parameters.

        Parameters:
        - x: The independent variable.
        - y: The dependent variable.
        - kwargs: Keyword arguments.
        """
        raise NotImplementedError

    @classmethod
    def fit(
        cls,
        x: _ARRAY,
        data: _ARRAY,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_t.Union[_T, tuple, list]] = None,
        method: _t.Literal["least_squares", "leastsq", "curve_fit"] = "leastsq",
        **kwargs,
    ) -> FitResult[_T]:  # Tuple[_T, _t.Callable, _NDARRAY]:
        """Fit the data using the specified fitting function.

        Parameters:
        - x: The independent variable.
        - data: The dependent variable.
        - mask: The mask array or threshold for data filtering (optional).
        - guess: The initial guess for fit parameters (optional).
        - kwargs: Additional keyword arguments.

        Returns:
        - FitResult: The result of the fit, including the fitted parameters and the fitted function.
        """
        x = np.asarray(x)
        data = np.asarray(data)

        mask = get_mask(mask, x)

        if np.sum(mask) < param_len(cls.param):
            return FitResult()
        x_masked = x[mask]
        data_masked = data[mask]

        if guess is None:
            guess = cls._guess(x_masked, data_masked, **kwargs)

        if method in {"least_squares", "leastsq"}:

            def to_minimize(args):
                return cls.func(x_masked, *args) - data_masked

            res, _ = optimize.leastsq(to_minimize, guess)
        elif method == "curve_fit":
            res, _ = optimize.curve_fit(
                cls.func,
                x_masked,
                data_masked,
                p0=guess,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        if cls.normalize_res is not None:  # type: ignore
            res = cls.normalize_res(res)  # type: ignore
        param = cls.param(*res)

        return FitResult(param, lambda x: cls.func(x, *res), x)

    @classmethod
    async def async_fit(
        cls,
        x: _ARRAY,
        data: _ARRAY,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_T] = None,
        **kwargs,
    ) -> FitResult[_T]:
        return cls.fit(x, data, mask, guess, **kwargs)

    @classmethod
    async def async_array_fit(
        cls,
        x: _ARRAY,
        data: _ARRAY,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_T] = None,
        **kwargs,
    ) -> FitArrayResult[_T]:
        tasks = [
            cls.async_fit(x, data[i], mask=mask, guess=guess, **kwargs) for i in range(len(data))
        ]
        results = await asyncio.gather(*tasks)

        def func(y):
            return np.array([res.res_func(y) for res in results])

        return FitArrayResult(results, func)

    @classmethod
    def array_fit(
        cls,
        x: _ARRAY,
        data: _ARRAY,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_T] = None,
        **kwargs,
    ) -> FitArrayResult[_T]:
        def func():
            return cls.async_array_fit(x, data, mask, guess, **kwargs)

        try:
            return asyncio.run(func())
        except RuntimeError as exc:
            raise RuntimeError(
                "asyncio.run() cannot be called from a running event loop."
                "Run ffit.nest_asyncio_apply() before calling this method."
            ) from exc

    # @classmethod
    # def sfit(
    #     cls,
    #     x: _ARRAY,
    #     data: _ARRAY,
    #     mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
    #     guess: _t.Optional[_T] = None,
    #     T: int = 1,
    #     **kwargs,
    # ) -> FitResult[_T]:
    #     """Fit the data using the specified fitting function with simulated annealing.

    #     Parameters:
    #     - x: The independent variable.
    #     - data: The dependent variable.
    #     - mask: The mask array or threshold for data filtering (optional).
    #     - guess: The initial guess for fit parameters (optional).
    #     - T: The temperature parameter for simulated annealing (default: 1).
    #     - kwargs: Additional keyword arguments.

    #     Returns:
    #     - FitResult: The result of the fit, including the fitted parameters and the fitted function.
    #     """
    #     mask = get_mask(mask, x)

    #     def to_minimize(args):
    #         return np.abs(np.sum((cls.func(x[mask], *args) - data[mask]) ** 2))

    #     if guess is None:
    #         guess = cls._guess(x[mask], data[mask], **kwargs)

    #     res = optimize.basinhopping(
    #         func=to_minimize,
    #         x0=guess,
    #         T=T,
    #         # minimizer_kwargs={"jac": lambda params: chisq_jac(sin_jac, x, y_data, params)}
    #     ).x

    #     return FitResult(cls.param(*res), lambda x: cls.func(x, *res))

    @classmethod
    def guess(
        cls,
        x,
        y,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_T] = None,
        **kwargs,
    ) -> FitResult[_T]:
        """Guess the initial fit parameters.

        Parameters:
        - x: The independent variable.
        - y: The dependent variable.
        - mask: The mask array or threshold for data filtering (optional).
        - kwargs: Additional keyword arguments.

        Returns:
        - Tuple[_T, _t.Callable]: The guessed fit parameters and the fitted function.
        """
        if guess is not None:
            return FitResult(cls.param(*guess), lambda x: cls.func(x, guess))
        mask = get_mask(mask, x)
        guess_param = cls._guess(x[mask], y[mask], **kwargs)
        return FitResult(cls.param(*guess_param), lambda x: cls.func(x, *guess_param))

    @classmethod
    def error(cls, func, x, y, **kwargs):
        """Calculate the error between the fitted function and the data.

        Parameters:
        - func: The fitted function.
        - x: The independent variable.
        - y: The dependent variable.
        - kwargs: Additional keyword arguments.

        Returns:
        - float: The error between the fitted function and the data.
        """
        del kwargs
        return np.sum(np.abs(func(x) - y) ** 2) / len(x)
