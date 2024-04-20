import abc
import re
import typing as _t

import jax.numpy as jnp

# import numpy as np
from scipy import optimize

_T = _t.TypeVar("_T")
_R = _t.TypeVar("_R")
if _t.TYPE_CHECKING:
    from matplotlib.axes import Axes


def param_len(cls):
    return len(cls.__annotations__)


_DEFAULT_COLORS: _t.Optional[_t.Dict[int, str]] = None


def get_color_by_int(index: int) -> _t.Optional[str]:
    global _DEFAULT_COLORS  # pylint: disable=W0603
    if _DEFAULT_COLORS is None:
        import matplotlib as mpl

        _DEFAULT_COLORS = dict(enumerate(mpl.rcParams["axes.prop_cycle"].by_key()["color"]))

    return _DEFAULT_COLORS.get(index % len(_DEFAULT_COLORS))


class FitResult(_t.Tuple[_t.Optional[_R], _t.Callable]):
    """
    Represents the result of a fit operation.

    Attributes:
        res: The result of the fit operation.
        res_func: A callable function that takes a numpy array as input and returns a numpy array as output.
    """

    res: _t.Optional[_R]
    res_func: _t.Callable[[jnp.ndarray], jnp.ndarray]
    x: _t.Optional[jnp.ndarray]

    def __init__(
        self,
        res: _t.Optional[_R] = None,
        res_func: _t.Optional[_t.Callable] = None,
        x: _t.Optional[jnp.ndarray] = None,
        **kwargs,
    ):
        """
        Initialize the Main class.

        Args:
            res: Optional result value.
            res_func: Optional callable function for result.
            **kwargs: Additional keyword arguments.
        """
        del kwargs
        self.res = res
        self.res_func = res_func if res_func is not None else (lambda x: jnp.ones_like(x) * jnp.nan)
        self.x = x

    def __new__(
        cls,
        res: _t.Optional[_R] = None,
        res_func: _t.Optional[_t.Callable] = None,
        x: _t.Optional[jnp.ndarray] = None,
        **kwargs,
    ):
        if res_func is None:
            res_func = lambda _: None  # noqa: E731

        new = super().__new__(cls, (res, res_func))
        return new

    def plot(
        self,
        ax: _t.Optional["Axes"],
        label: str = "Fit",
        color: _t.Optional[_t.Union[str, int]] = None,
        title: _t.Optional[str] = None,
        **kwargs,
    ):
        if ax is None:
            import matplotlib.pyplot as plt
            from matplotlib.axes import Axes

            ax = plt.gca()  # type: ignore
            if not isinstance(ax, Axes):
                raise ValueError("Axes cannot be get from plt.gca. It must be provided.")

        if ax is None:
            raise ValueError("Axes must be provided.")
        if self.x is None:
            lims = ax.get_xlim()
            x_fit = jnp.linspace(*lims, 200)
        elif len(self.x) < 100:
            x_fit = jnp.linspace(jnp.min(self.x), jnp.max(self.x), 200)
        else:
            x_fit = self.x
        y_fit = self.res_func(x_fit)

        label = self._format_str_with_params(label)
        if isinstance(color, int) or (
            isinstance(color, str) and color.isdigit() and len(color) == 1
        ):
            color = get_color_by_int(int(color))

        ax.plot(x_fit, y_fit, label=label, color=color, **kwargs)

        if title:
            title = self._format_str_with_params(title)
            current_title = ax.get_title()
            if current_title:
                title = f"{current_title}\n{title}"
            ax.set_title(title)

        return self

    def _format_str_with_params(self, text: str, default_precision: str = ".2f"):
        if self.res is None or "$" not in text:
            return text

        possible_params = re.findall(r"\$(\d)(\.\d[fed])?", text)
        if not possible_params:
            return text
        for index, precision in possible_params:
            index = int(index)
            if index is None or index >= len(self.res):  # type: ignore
                continue
            if precision is None:
                precision = default_precision
                to_replace = f"${index}"
            else:
                to_replace = f"${index}{precision}"

            param = self.res[index]  # type: ignore
            text = text.replace(to_replace, f"{format(param, precision)}")

        return text


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

    func: _t.Callable[..., jnp.ndarray]
    jac: _t.Optional[_t.Callable[..., jnp.ndarray]]

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
        x: jnp.ndarray,
        data: jnp.ndarray,
        mask: _t.Optional[_t.Union[jnp.ndarray, float]] = None,
        guess: _t.Optional[_T] = None,
        **kwargs,
    ) -> FitResult[_T]:  # Tuple[_T, _t.Callable, jnp.ndarray]:
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
        x = jnp.asarray(x)
        data = jnp.asarray(data)

        mask = cls.get_mask(mask, x)

        if jnp.sum(mask) < param_len(cls.param):
            return FitResult()

        def to_minimize(args):
            return jnp.abs(cls.func(x[mask], args) - data[mask])

        if guess is None:
            guess = cls._guess(x[mask], data[mask], **kwargs)
        # if self.jac is not None:
        #     res = optimize.leastsq(to_minimize, guess, Dfun=self.jac, full_output=True, **kwargs)
        res, _ = optimize.leastsq(to_minimize, guess, maxfev=5000)  # full_output=True)
        # res, _, infodict, _, _ = leastsq(to_minimize, guess, full_output=True)

        return FitResult(cls.param(*res), lambda x: cls.func(x, res), x)

    @classmethod
    async def async_fit(
        cls,
        x: jnp.ndarray,
        data: jnp.ndarray,
        mask: _t.Optional[_t.Union[jnp.ndarray, float]] = None,
        guess: _t.Optional[_T] = None,
        **kwargs,
    ) -> FitResult[_T]:
        return cls.fit(x, data, mask, guess, **kwargs)

    @classmethod
    def sfit(
        cls,
        x: jnp.ndarray,
        data: jnp.ndarray,
        mask: _t.Optional[_t.Union[jnp.ndarray, float]] = None,
        guess: _t.Optional[_T] = None,
        T: int = 1,
        **kwargs,
    ) -> FitResult[_T]:
        """Fit the data using the specified fitting function with simulated annealing.

        Parameters:
        - x: The independent variable.
        - data: The dependent variable.
        - mask: The mask array or threshold for data filtering (optional).
        - guess: The initial guess for fit parameters (optional).
        - T: The temperature parameter for simulated annealing (default: 1).
        - kwargs: Additional keyword arguments.

        Returns:
        - FitResult: The result of the fit, including the fitted parameters and the fitted function.
        """
        mask = cls.get_mask(mask, x)

        def to_minimize(args):
            return jnp.abs(jnp.sum((cls.func(x[mask], *args) - data[mask]) ** 2))

        if guess is None:
            guess = cls._guess(x[mask], data[mask], **kwargs)

        res = optimize.basinhopping(
            func=to_minimize,
            x0=guess,
            T=T,
            # minimizer_kwargs={"jac": lambda params: chisq_jac(sin_jac, x, y_data, params)}
        ).x

        return FitResult(cls.param(*res), lambda x: cls.func(x, *res))

    @classmethod
    def guess(
        cls, x, y, mask: _t.Optional[_t.Union[jnp.ndarray, float]] = None, **kwargs
    ) -> _t.Tuple[_T, _t.Callable]:
        """Guess the initial fit parameters.

        Parameters:
        - x: The independent variable.
        - y: The dependent variable.
        - mask: The mask array or threshold for data filtering (optional).
        - kwargs: Additional keyword arguments.

        Returns:
        - Tuple[_T, _t.Callable]: The guessed fit parameters and the fitted function.
        """
        mask = cls.get_mask(mask, x)
        guess_param = cls._guess(x[mask], y[mask], **kwargs)
        return cls.param(*guess_param), lambda x: cls.func(x, *guess_param)

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
        return jnp.sum(jnp.abs(func(x) - y) ** 2) / len(x)

    @staticmethod
    def get_mask(
        mask: _t.Optional[_t.Union[jnp.ndarray, float, list]] = None,
        x: _t.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Return a mask array based on the provided mask or threshold.

        Parameters:
        - mask: The mask array or threshold (optional).
        - x: The independent variable (optional).
        Returns:

        - jnp.ndarray: The mask array.
        """
        if mask is None:
            if x is None:
                raise ValueError("Either x or mask must be provided.")
            return jnp.ones_like(x, dtype=bool)
        elif isinstance(mask, (int, float)):
            if x is None:
                raise ValueError("Mask cannot be float if x is not provided.")
            return x < mask
        return jnp.array(mask)
