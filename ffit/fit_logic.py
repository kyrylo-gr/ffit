import asyncio
import typing as _t

import numpy as np
from scipy import optimize

from .fit_results import FitResult
from .utils import (
    _2DARRAY,
    _ARRAY,
    _NDARRAY,
    FuncParamProtocol,
    classproperty,
    get_masked_data,
    get_random_subarrays,
    mask_func,
    mask_func_result,
    std_monte_carlo,
)

_T = _t.TypeVar("_T", bound=FitResult)
_POSSIBLE_FIT_METHODS = _t.Literal["least_squares", "leastsq", "curve_fit"]


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

    _result_class: _t.Type[_T]  # abc.ABCMeta

    def __init__(self, *args, **kwargs):
        """
        Initialize the FitLogic instance.

        Parameters:
        - args: Positional arguments.
        - kwargs: Keyword arguments.
        """
        del args
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)

        self._param_len = len(self._result_class.keys)

    func: _t.Callable[..., _NDARRAY]
    func_std: _t.Callable[..., _NDARRAY]
    normalize_res: _t.Optional[_t.Callable[[_t.Sequence[float]], _NDARRAY]] = None
    jac: _t.Optional[_t.Callable[..., _NDARRAY]] = None

    # Parameters to generate example figure for the documentation
    _example_param: tuple
    _example_x_min: float
    _example_x_max: float
    _example_x_points: int
    _example_std: float

    # Parameters for the tests:
    _range_x: _t.Tuple[float, float]

    # Additional attributes:
    _test_ignore: bool  # Ignore generic tests
    _doc_ignore: bool  # Ignore automatic documentation generation
    _doc_list_ignore: bool  # Ignore this function in the list of functions. However, add a page for it.

    def get_func_std(self):
        return getattr(self, "func_std", self.default_func_std)

    def default_func_std(
        self,
        x: _NDARRAY,
        *args,
        n_simulations: int = 10_000,
    ):
        half_index = len(args) // 2
        mean_args = args[:half_index]
        std_args = args[half_index:]

        return std_monte_carlo(
            x, self.func, mean_args, std_args, n_simulations=n_simulations
        )

    @classproperty
    def param(self) -> FuncParamProtocol:
        return self._result_class.param_class  # type: ignore

    @staticmethod
    def _guess(x, y, **kwargs):
        """Abstract method for guessing initial fit parameters.

        Parameters:
        - x: The independent variable.
        - y: The dependent variable.
        - kwargs: Keyword arguments.
        """
        raise NotImplementedError

    def fit(
        self,
        x: _ARRAY,
        data: _ARRAY,
        *,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_t.Union[_T, tuple, list]] = None,
        method: _t.Literal["least_squares", "leastsq", "curve_fit"] = "leastsq",
        maxfev: int = 10000,
        **kwargs,
    ) -> _T:  # Tuple[_T, _t.Callable, _NDARRAY]:
        """
        Fit the data using the specified fitting function.

        This function returns [FitResult][ffit.fit_results.FitResult] see
        the documentation for more information what is possible with it.


        Args:
            x: The independent variable.
            data: The dependent variable.
            mask: The mask array or threshold for data filtering (optional).
            guess: The initial guess for fit parameters (optional).
            method: The fitting method to use. Valid options are "least_squares", "leastsq",
                and "curve_fit" (default: "leastsq").
            **kwargs: Additional keyword arguments.

        Returns:
            FitResult: The result of the fit, including the fitted parameters and the fitted function.

        Raises:
            ValueError: If an invalid fitting method is provided.

        """
        # Convert x and data to numpy arrays
        x, data = np.asarray(x), np.asarray(data)

        res, res_std = self._fit(
            x,
            data,
            mask=mask,
            guess=guess,
            method=method,
            maxfev=maxfev,
            **kwargs,
        )

        # param = self.param(*res, std=res_std)

        full_func = getattr(self, "full_func", self.__class__().func)

        # print(res)
        return self._result_class(
            res,
            lambda x: full_func(x, *res),
            std=res_std,
            x=x,
            data=data,
            stderr=res_std,
            stdfunc=lambda x: self.get_func_std()(x, *res, *res_std),
            original_func=self.func,
        )

    def _fit(
        self,
        x: _NDARRAY,
        data: _NDARRAY,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_t.Union[_T, tuple, list]] = None,
        method: _t.Literal["least_squares", "leastsq", "curve_fit"] = "leastsq",
        maxfev: int = 10000,
        **kwargs,
    ) -> _t.Tuple[np.ndarray, np.ndarray]:

        # Mask the data and check that length of masked data is greater than lens of params
        x_masked, data_masked = get_masked_data(x, data, mask, self._param_len)
        if len(x_masked) == 0 or len(data_masked) == 0:
            return (
                np.ones(self._param_len) * np.nan,
                np.ones(self._param_len) * np.nan,
            )

        # Get a guess if not provided
        if guess is None:
            guess = self._guess(x_masked, data_masked, **kwargs)

        guess = tuple(guess)  # type: ignore
        # Fit the data
        res, cov = self._run_fit(x_masked, data_masked, guess, method, maxfev=maxfev)

        # Normalize the result if necessary. Like some periodicity that should not be important
        if self.normalize_res is not None:  # type: ignore
            res = self.normalize_res(res)  # type: ignore

        # Convert the result to a parameter object (NamedTuple)
        # print(cov)
        if cov is not None:
            stds = np.diag(cov)
            if self.normalize_res is not None:  # type: ignore
                stds = self.normalize_res(stds)  # type: ignore
            param_std = stds
        else:
            param_std = np.ones_like(res) * np.nan
        return res, param_std

    async def _async_fit(self, *args, **kwargs) -> _NDARRAY:
        return self._fit(*args, **kwargs)[0]

    async def async_fit(
        self,
        x: _ARRAY,
        data: _ARRAY,
        *,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_T] = None,
        method: _t.Literal["least_squares", "leastsq", "curve_fit"] = "leastsq",
        maxfev: int = 10000,
        **kwargs,
    ) -> _T:
        """
        Asynchronously fits the model to the provided data.

        Args:
            x (_ARRAY): The independent variable data.
            data (_ARRAY): The dependent variable data to fit.
            mask (Optional[Union[_ARRAY, float]], optional): An optional mask to apply to the data. Defaults to None.
            guess (Optional[_T], optional): An optional initial guess for the fitting parameters. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the fitting function.

        Returns:
            FitWithErrorResult[_T]:
                The result of the fitting process, including the fitted parameters and associated errors.
        """
        return self.fit(
            x, data, mask=mask, guess=guess, method=method, maxfev=maxfev, **kwargs
        )

    async def async_array_fit(
        self,
        x: _ARRAY,
        data: _2DARRAY,
        *,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_t.Union[_T, _ARRAY]] = None,
        axis: int = -1,
        method: _t.Literal["least_squares", "leastsq", "curve_fit"] = "leastsq",
        maxfev: int = 10000,
        **kwargs,
    ) -> _T:

        # Convert x and data to numpy arrays
        x, data = np.asarray(x), np.asarray(data)
        if axis != -1:
            data = np.moveaxis(data, axis, -1)
        data_shape = data.shape
        selected_axis_len = data_shape[-1]
        data = data.reshape(-1, selected_axis_len)

        tasks = [
            self._async_fit(
                x,
                data[i],
                mask=mask,
                guess=guess,
                method=method,
                maxfev=maxfev,
                **kwargs,
            )
            for i in range(len(data))
        ]
        results = await asyncio.gather(*tasks)
        results = np.array(results)
        fit_param_len = results.shape[-1]

        full_func = getattr(self, "full_func", self.__class__().func)
        results = results.reshape(data_shape[:-1] + (-1,))

        def res_func(xx):
            return np.array(
                [full_func(xx, *res) for res in results.reshape(-1, fit_param_len)]
            ).reshape(data_shape[:-1] + (-1,))

        return self._result_class(
            results,
            res_func,
            x=x,
            data=data,
            original_func=self.func,
        )

    def array_fit(
        self,
        x: _ARRAY,
        data: _2DARRAY,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_T] = None,
        **kwargs,
    ) -> _T:
        async def func():
            return await self.async_array_fit(x, data, mask=mask, guess=guess, **kwargs)

        try:
            return asyncio.run(func())  # type: ignore
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
        data,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_T] = None,
        **kwargs,
    ) -> _T:
        """Guess the initial fit parameters.

        This function returns an object of the class `FitResult`.
        See its documentation for more information on what is possible with it.

        Args:
            x: The independent variable.
            data: The dependent variable.
            mask: The mask array or threshold for data filtering (optional).
            guess: The initial guess for the fit parameters (optional).
            **kwargs: Additional keyword arguments.

        Returns:
            FitResult: The guess, including the guess parameters and the function based on the guess.

        Examples:
            >>> x = [1, 2, 3, 4, 5]
            >>> data = [2, 4, 6, 8, 10]
            >>> fit_guess = FitLogic.guess(x, data)
            >>> fit_guess.plot()
        """
        if guess is not None:
            return cls._result_class(
                np.asarray(guess),
                lambda x: cls.func(x, *guess),
                x=x,
                data=data,
            )
        x_masked, data_masked = get_masked_data(x, data, mask, mask_min_len=1)
        guess_param = cls._guess(x_masked, data_masked, **kwargs)
        return cls._result_class(
            np.asarray(guess_param),
            lambda x: cls.func(x, *guess_param),
            x=x,
            data=data,
        )

    # @classmethod
    # def error(cls, func, x, y, **kwargs):
    #     """Calculate the error between the fitted function and the data.

    #     Parameters:
    #     - func: The fitted function.
    #     - x: The independent variable.
    #     - y: The dependent variable.
    #     - kwargs: Additional keyword arguments.

    #     Returns:
    #     - float: The error between the fitted function and the data.
    #     """
    #     del kwargs
    #     return np.sum(np.abs(func(x) - y) ** 2) / len(x)

    def bootstrapping(
        self,
        x: _ARRAY,
        data: _ARRAY,
        mask: _t.Optional[_t.Union[_ARRAY, float]] = None,
        guess: _t.Optional[_t.Union[_T, tuple, list]] = None,
        method: _t.Literal["least_squares", "leastsq", "curve_fit"] = "leastsq",
        num_of_permutations: _t.Optional[int] = None,
        **kwargs,
    ) -> _T:  # Tuple[_T, _t.Callable, _NDARRAY]:
        """
        Fit the data using the specified fitting function.

        This function returns [FitResult][ffit.fit_results.FitResult] see
        the documentation for more information what is possible with it.

        Args:
            x: The independent variable.
            data: The dependent variable.
            mask: The mask array or threshold for data filtering (optional).
            guess: The initial guess for fit parameters (optional).
            method: The fitting method to use. Valid options are "least_squares", "leastsq",
                and "curve_fit" (default: "leastsq").
            **kwargs: Additional keyword arguments.

        Returns:
            FitResult: The result of the fit, including the fitted parameters and the fitted function.

        Raises:
            ValueError: If an invalid fitting method is provided.

        """
        # Convert x and data to numpy arrays
        x, data = np.asarray(x), np.asarray(data)

        # Mask the data and check that length of masked data is greater than lens of params
        x_masked, data_masked = get_masked_data(x, data, mask, self._param_len)
        if len(x_masked) == 0 or len(data_masked) == 0:
            return self._result_class(
                np.ones(self._param_len) * np.nan,
                x=x,
                data=data,
            )

        # Get a guess if not provided
        if guess is None:
            guess = self._guess(x_masked, data_masked, **kwargs)

        # Fit ones to get the best initial guess
        guess, cov = self._run_fit(x_masked, data_masked, guess, method)

        # Run fit on random subarrays
        all_res = []
        for xx, yy in get_random_subarrays(x_masked, data_masked, num_of_permutations):
            res, _ = self._run_fit(xx, yy, guess, method)
            if self.normalize_res is not None:  # type: ignore
                res = self.normalize_res(res)  # type: ignore
            all_res.append(res)

        res_means = np.mean(all_res, axis=0)
        bootstrap_std = np.std(all_res, axis=0)
        # total_std = np.sqrt(np.diag(cov) + bootstrap_std**2)
        total_std = bootstrap_std

        full_func = getattr(self, "full_func", self.__class__().func)

        return self._result_class(
            res_means,
            lambda x: full_func(x, *res_means),
            x=x,
            data=data,
            cov=cov,  # type: ignore
            stderr=total_std,  # type: ignore
            stdfunc=lambda x: self.get_func_std()(x, *res_means, *total_std),
            original_func=self.func,
        )

    def _run_fit(self, x, y, guess, method: _POSSIBLE_FIT_METHODS, maxfev: int = 10000):
        if method in {"least_squares", "leastsq"}:

            def to_minimize(args):
                return np.abs(self.func(x, *args) - y)

            # opt, cov, infodict, msg, ier
            opt, cov, _, _, _ = optimize.leastsq(  # type: ignore
                to_minimize, guess, full_output=True, maxfev=maxfev
            )

            return opt, cov
        elif method == "curve_fit":
            raise ValueError(f"Invalid method: {method}")

            # res, _ = optimize.curve_fit(
            #     cls.func,
            #     x_masked,
            #     data_masked,
            #     p0=guess,
            #     **kwargs,
            # )
        else:
            raise ValueError(f"Invalid method: {method}")

    @classmethod
    def mask(cls, **kwargs):
        instance = cls()

        params_fields = cls._result_class.keys
        mask = np.ones(len(params_fields)).astype(bool)
        mask_values = np.zeros(len(params_fields))
        for param_name, param_value in kwargs.items():
            for i, possible_name in enumerate(params_fields):
                if param_name == possible_name:
                    mask[i] = False
                    mask_values[i] = param_value

        # transparent_mask = len(mask) == np.count_nonzero(mask)

        default_func = instance.func
        instance.func = staticmethod(mask_func(default_func, mask, mask_values))

        default_guess = instance._guess
        instance._guess = staticmethod(mask_func_result(default_guess, mask))

        default_normalize_res = instance.normalize_res  # type: ignore
        if default_normalize_res is None:

            def default_normalize_res(x):
                return x

        def masked_normize_res(x):
            # print(mask, np.count_nonzero(mask), x)
            if len(mask) == len(x):
                x = np.array(x)[mask]
            input_full = np.zeros_like(mask).astype(float)
            input_full[mask] = x
            input_full[~mask] = mask_values[~mask]
            res_full = np.array(default_normalize_res(input_full))
            return res_full

        instance.normalize_res = staticmethod(masked_normize_res)

        return instance
