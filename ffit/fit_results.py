import typing as _t

import numpy as np

from .config import DEFAULT_FIT_LABEL
from .utils import _NDARRAY, format_str_with_params, get_right_color

_R = _t.TypeVar("_R", bound=_t.Sequence)
if _t.TYPE_CHECKING:
    from matplotlib.axes import Axes


def get_ax_from_gca(ax: _t.Optional["Axes"] = None) -> "Axes":
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    ax = plt.gca()  # type: ignore
    if not isinstance(ax, Axes):
        raise ValueError("Axes cannot be get from plt.gca. It must be provided.")
    return ax


def get_x_from_ax(ax: "Axes", expected_len: _t.Optional[int] = None) -> _NDARRAY:
    lines = ax.get_lines()
    if len(lines) == 0:
        raise ValueError("No lines found in the plot. X must be provided.")
    line = lines[0]
    if hasattr(line, "get_xdata"):
        x = line.get_xdata(orig=True)
        assert isinstance(x, _t.Iterable)
        if expected_len and len(x) != expected_len:
            raise ValueError("X must be provided. Cannot be extracted from the plot.")
        return np.array(x)
    raise ValueError("X must be provided.")


def create_x_from_ax(
    ax: "Axes", x: _t.Optional[_NDARRAY] = None, points: int = 200
) -> _NDARRAY:
    if x is None:
        lims = ax.get_xlim()
        return np.linspace(*lims, points)
    if len(x) < 100:
        return np.linspace(np.min(x), np.max(x), points)
    return x


def get_right_x(
    x: _t.Optional[_t.Union[_NDARRAY, int]],
    ax: "Axes",
    possible_x: _t.Optional[_NDARRAY],
) -> _NDARRAY:
    if x is None:
        return create_x_from_ax(ax, possible_x)
    if isinstance(x, int):
        return create_x_from_ax(ax, possible_x, points=x)
    return x


class FitResult(_t.Tuple[_t.Optional[_R], _t.Callable]):
    """This class represents the result of a fit operation.

    Examples
    --------
        >>> import ffit as ff
        >>> result = ff.Cos().fit(x, y)
        >>> result.res.amplitude # to get the amplitude
        >>> result.res # to get whole result as a NamedTuple
        >>> y0 = result.res_func(x0) # to get the fitted values
        >>> result.plot() # to plot the fit results

        All in one:
        >>> amp = ff.Cos().fit(x, y).plot().res.amplitude

    Unpack the FitResult
    ----------------------
    The FitResult is based on a tuple with two elements to have right type hints.
    You can unpack the FitResult like this:

        >>> res, res_func = ff.Cos().fit(x, y)

    You can also unpack it after using some methods:

        >>> res, res_func = ff.Cos().fit(x, y).plot(ax, x=x, label="Cosine fit")

    Remember: fit method always returns FitResult that can be unpacked.
    In case the fit fails, the FitResult will have `res=None, res_func=lambda x: [np.nan] * len(x)`.


    Remember: The FitResult is immutable as based on the tuple, so you cannot change the values.
    """

    res: _t.Optional[_R]
    res_func: _t.Callable[[_NDARRAY], _NDARRAY]
    x: _t.Optional[_NDARRAY]
    data: _t.Optional[_NDARRAY]
    cov: _t.Optional[_NDARRAY]

    def __init__(
        self,
        res: _t.Optional[_R] = None,
        res_func: _t.Optional[_t.Callable] = None,
        x: _t.Optional[_NDARRAY] = None,
        data: _t.Optional[_NDARRAY] = None,
        cov: _t.Optional[_NDARRAY] = None,
        **kwargs,
    ):
        """
        Initialize the FitResult class.
        ---------------------------

        Args:
            res: Result value as NamedTuple.
            res_func: Optional callable function for result.
            x: Original x values used to fitted.
            data: Original data that was fitted.
            **kwargs: Additional keyword arguments that will be ignored.

        Example to create yourself.
        -----------------------------
            >>> result = ff.FitResult(res=(1, 2, 3), res_func=lambda x: x ** 2)

        """
        del kwargs
        self.res = res
        self.res_func = (
            res_func if res_func is not None else (lambda x: np.ones_like(x) * np.nan)
        )
        self.x = x
        self.data = data
        self.cov = cov

    def __new__(
        cls,
        res: _t.Optional[_R] = None,
        res_func: _t.Optional[_t.Callable] = None,
        x: _t.Optional[_NDARRAY] = None,
        **kwargs,
    ):
        if res_func is None:
            res_func = lambda _: None  # noqa: E731

        new = super().__new__(cls, (res, res_func))
        return new

    def plot(
        self,
        ax: _t.Optional["Axes"] = None,
        *,
        x: _t.Optional[_t.Union[_NDARRAY, int]] = None,
        label: _t.Optional[str] = DEFAULT_FIT_LABEL,
        color: _t.Optional[_t.Union[str, int]] = None,
        title: _t.Optional[str] = None,
        post_func_x: _t.Optional[_t.Callable[[_NDARRAY], _NDARRAY]] = None,
        post_func_y: _t.Optional[_t.Callable[[_NDARRAY], _NDARRAY]] = None,
        **kwargs,
    ):
        """Plot the fit results on the given axes.

        Args:
            ax (Optional[Axes]): The axes on which to plot the fit results. If None, a new axes will be created.
            label (str): The label for the plot. Defaults to ffit.config.DEFAULT_FIT_LABEL.
            color (Optional[Union[str, int]]): The color of the plot. If None, a default color will be used.
            title (Optional[str]): The title for the plot. If provided, it will be appended to the existing title.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            FitResults: The FitResults object itself.

        Example:
            ```
            >>> result = ff.Cos().fit(x, y)
            >>> result.plot() # ax will be get from plt.gca()
            >>> result.plot(ax, x=x, label="Cosine fit")
            >>> result.plot(ax, x=x, label="Cosine fit", color="r")
            ```
        Worth to mention: title will be appended to the existing title with a new line.


        """
        ax = get_ax_from_gca(ax)
        x_fit = get_right_x(x, ax, self.x)

        y_fit = self.res_func(x_fit)
        if label is not None:
            label = format_str_with_params(self.res, label)
            kwargs.update({"label": label})

        color = get_right_color(color)
        kwargs.update({"color": color})

        if post_func_x:
            x_fit = post_func_x(x_fit)
        if post_func_y:
            y_fit = post_func_y(y_fit)
        ax.plot(x_fit, y_fit, **kwargs)

        if title:
            title = format_str_with_params(self.res, title)
            current_title = ax.get_title()
            if current_title:
                title = f"{current_title}\n{title}"
            ax.set_title(title)

        if label != DEFAULT_FIT_LABEL and label is not None:
            ax.legend()

        return self


class FitArrayResult(_t.Tuple[_t.List[_t.Optional[_R]], _t.Callable]):
    res: _t.Optional[_t.List[FitResult[_R]]]
    res_func: _t.Callable[[_NDARRAY], _NDARRAY]
    x: _t.Optional[_NDARRAY]
    extracted_data: _t.Dict[str, _NDARRAY]

    def __init__(
        self,
        res: _t.Optional[_t.List[FitResult[_R]]] = None,
        res_func: _t.Optional[_t.Callable] = None,
        x: _t.Optional[_NDARRAY] = None,
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
        self.res_func = (
            res_func if res_func is not None else (lambda x: np.ones_like(x) * np.nan)
        )
        self.x = x
        self.extracted_data = {}

    def __new__(
        cls,
        res: _t.Optional[_t.List[FitResult[_R]]] = None,
        res_func: _t.Optional[_t.Callable] = None,
        x: _t.Optional[_NDARRAY] = None,
        **kwargs,
    ):

        if res_func is None:
            res_func = lambda x: np.ones_like(x) * np.nan  # noqa: E731

        new = super().__new__(cls, (res, res_func))  # type: ignore
        return new

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)

        if self.extracted_data is None:
            raise KeyError("No functions have been set.")
        if key not in self.extracted_data:
            raise KeyError(f"Function with key {key} not found.")
        return self.extracted_data[key]

    # def extract(self, parameter: _t.Union[str, int], data_name: _t.Optional[str] = None):
    #     if data_name is None:
    #         data_name = str(parameter)

    #     self.extracted_data[data_name] = self.get(parameter)
    #     return self

    def get(self, parameter: _t.Union[str, int]) -> _NDARRAY:
        if self.res is None:
            raise ValueError("No results have been set.")

        def get_key(res: FitResult[_R]):
            if res.res is None:
                return np.nan
            if isinstance(parameter, int):
                return res.res[parameter]  # type: ignore
            return getattr(res.res, parameter)

        return np.array([get_key(f) for f in self.res])

    def plot(
        self,
        ax: _t.Optional["Axes"] = None,
        *,
        x: _t.Optional[_NDARRAY] = None,
        data: _t.Optional[_t.Union[str, int]] = None,
        label: str = DEFAULT_FIT_LABEL,
        color: _t.Optional[_t.Union[str, int]] = None,
        title: _t.Optional[str] = None,
        **kwargs,
    ):
        if ax is None:
            ax = get_ax_from_gca()
        if data is None:
            raise ValueError("Data must be provided.")
        y_fit = self.get(data)

        if x is None:
            x = get_x_from_ax(ax, len(y_fit))

        if label != DEFAULT_FIT_LABEL or kwargs.get("legend", False):
            ax.legend()
        if label == DEFAULT_FIT_LABEL:
            if isinstance(data, int):
                try:
                    data = self.res[0].res._fields[data]  # type: ignore
                except AttributeError:
                    data = str(data)
            label = f"Fit of {data}"

        color = get_right_color(color)

        ax.plot(x, y_fit, label=label, color=color, **kwargs)

        if title:
            current_title = ax.get_title()
            if current_title:
                title = f"{current_title}\n{title}"
            ax.set_title(title)

        return self


class FitWithErrorResult(FitResult[_R]):
    res: _t.Optional[_R]
    stderr: _NDARRAY
    stdfunc: _t.Callable[[_NDARRAY], _NDARRAY]

    def __init__(
        self,
        res: _t.Optional[_R] = None,
        res_func: _t.Optional[_t.Callable] = None,
        x: _t.Optional[_NDARRAY] = None,
        data: _t.Optional[_NDARRAY] = None,
        cov: _t.Optional[_NDARRAY] = None,
        stderr: _t.Optional[_NDARRAY] = None,
        stdfunc: _t.Optional[_t.Callable] = None,
        **kwargs,
    ):
        """
        Initialize the FitResult class.
        ---------------------------

        Args:
            res: Result value as NamedTuple.
            res_func: Optional callable function for result.
            x: Original x values used to fitted.
            data: Original data that was fitted.
            **kwargs: Additional keyword arguments that will be ignored.

        Example to create yourself.
        -----------------------------
            >>> result = ff.FitResult(res=(1, 2, 3), res_func=lambda x: x ** 2)

        """
        super().__init__(res=res, res_func=res_func, x=x, data=data, cov=cov, **kwargs)
        self.stderr = stderr if stderr is not None else np.zeros_like(res)
        self.stdfunc = stdfunc if stdfunc is not None else np.zeros_like

    def plot_thick(
        self,
        ax: _t.Optional["Axes"] = None,
        *,
        x: _t.Optional[_t.Union[_NDARRAY, int]] = None,
        label: str = DEFAULT_FIT_LABEL,
        color: _t.Optional[_t.Union[str, int]] = None,
        title: _t.Optional[str] = None,
        kwargs_fill: _t.Optional[_t.Dict[str, _t.Any]] = None,
        **kwargs,
    ):
        """Plot the fit results on the given axes.

        Args:
            ax (Optional[Axes]): The axes on which to plot the fit results. If None, a new axes will be created.
            label (str): The label for the plot. Defaults to ffit.config.DEFAULT_FIT_LABEL.
            color (Optional[Union[str, int]]): The color of the plot. If None, a default color will be used.
            title (Optional[str]): The title for the plot. If provided, it will be appended to the existing title.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            FitResults: The FitResults object itself.

        Example:
            ```
            >>> result = ff.Cos().fit(x, y)
            >>> result.plot() # ax will be get from plt.gca()
            >>> result.plot(ax, x=x, label="Cosine fit")
            >>> result.plot(ax, x=x, label="Cosine fit", color="r")
            ```
        Worth to mention: title will be appended to the existing title with a new line.


        """
        ax = get_ax_from_gca()
        x_fit = get_right_x(x, ax, self.x)

        y_fit = self.res_func(x_fit)
        y_std = self.stdfunc(x_fit)

        y_1 = y_fit - y_std
        y_2 = y_fit + y_std

        label = format_str_with_params(self.res, label)

        color = get_right_color(color)
        # return x_fit, y_1, y_2

        # ax.plot(x_fit, y_fit, label=label, color=color, **kwargs)

        kwargs_fill = kwargs_fill or {}
        kwargs_fill.setdefault("color", color)
        ax.fill_between(x_fit, y_1, y_2, **kwargs_fill)  # type: ignore
        kwargs.setdefault("ls", "--")
        ax.plot(x_fit, np.mean([y_1, y_2], axis=0), label=label, color=color, **kwargs)

        if title:
            title = format_str_with_params(self.res, title)
            current_title = ax.get_title()
            if current_title:
                title = f"{current_title}\n{title}"
            ax.set_title(title)

        if label != DEFAULT_FIT_LABEL:
            ax.legend()

        return self
