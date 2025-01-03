import typing as _t

import numpy as np

from .config import DEFAULT_FIT_LABEL
from .utils import (
    _NDARRAY,
    FuncParamClass,
    convert_param_class,
    convert_to_label_instance,
    get_right_color,
)

_T = _t.TypeVar("_T")

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


class FitResult(_t.Generic[_T]):
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

    """

    res_array: _NDARRAY
    keys: _t.Tuple[str, ...]
    res_func: _t.Callable[[_NDARRAY], _NDARRAY]
    x: _t.Optional[_NDARRAY]
    data: _t.Optional[_NDARRAY]
    cov: _t.Optional[_NDARRAY]

    stderr: _NDARRAY
    stdfunc: _t.Callable[[_NDARRAY], _NDARRAY]

    param_class: _t.Type

    success: bool
    _ndim: int

    def __init__(
        self,
        res: _NDARRAY,
        res_func: _t.Optional[_t.Callable] = None,
        x: _t.Optional[_NDARRAY] = None,
        data: _t.Optional[_NDARRAY] = None,
        cov: _t.Optional[_NDARRAY] = None,
        std: _t.Optional[_NDARRAY] = None,
        stderr: _t.Optional[_NDARRAY] = None,
        stdfunc: _t.Optional[_t.Callable] = None,
        keys: _t.Optional[_t.Tuple[str, ...]] = None,
        original_func: _t.Optional[_t.Callable] = None,
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
        self.res_array = np.asarray(res)
        self._ndim = self.res_array.ndim
        self.res_func = (
            res_func if res_func is not None else (lambda x: np.ones_like(x) * np.nan)
        )
        self.x = x
        self.data = data
        self.cov = cov

        if std is None:
            std = np.ones_like(res) * np.nan
        self._std_array = std

        self.stderr = stderr if stderr is not None else np.zeros_like(res)
        self.stdfunc = (
            stdfunc if stdfunc is not None else (lambda x: np.ones_like(x) * np.nan)
        )
        self._res_dict = {}

        self.success = bool(np.all(np.isnan(self.res_array)))

        if keys is not None:
            self.keys = keys
        self._original_func = original_func

        if not hasattr(self.__class__, "param_class"):

            class RecreatedParamClass(FuncParamClass):
                keys = self.keys

            self.param_class = convert_param_class(RecreatedParamClass)

    def get(self, parameter: _t.Union[str, int]) -> _NDARRAY:
        if isinstance(parameter, int):
            return self.res_array[..., parameter]

        if parameter not in self.keys:
            raise ValueError(f"Parameter {parameter} not found.")
        if parameter in self._res_dict:
            return self._res_dict[parameter]
        if self._ndim > 1:
            val = self.res_array[..., self.keys.index(parameter)]
        else:
            val = self.res_array[self.keys.index(parameter)]
        self._res_dict[parameter] = val
        return val

    def get_result_at(self, index: int):
        if self._ndim > 1:
            res = self.res_array[index]
        else:
            res = self.res_array

        return self.__class__(
            res,
            lambda xx: self._original_func(xx, *res) if self._original_func else None,
            x=self.x,
            data=self.data,
            cov=self.cov,
            std=self._std_array,
            stderr=self.stderr,
            stdfunc=self.stdfunc,
            keys=self.keys,
            original_func=self._original_func,
        )

    def __getattr__(self, name: str) -> _t.Any:
        return self.get(name)

    def res_and_func(self) -> _t.Tuple[_NDARRAY, _t.Callable]:
        return self.res_array, self.res_func

    @property
    def res(self) -> _T:
        return self.param_class(*self.res_array)  # type: ignore

    @property
    def label(self) -> _T:
        return convert_to_label_instance(self.param_class, self.res_array)  # type: ignore

    @property
    def std(self) -> _T:
        return self.param_class(*self._std_array)  # type: ignore

    def asdict(self) -> _t.Dict[str, _NDARRAY]:
        return {key: self.get(key) for key in self.keys}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.keys})"

    def __iter__(self):
        return iter(self.res_array)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.get(index)
        if isinstance(index, int):
            return self.res_array[..., index]
        return self.res_array[index]

    def plot(
        self,
        ax: _t.Optional["Axes"] = None,
        *,
        x: _t.Optional[_t.Union[_NDARRAY, int]] = None,
        label: _t.Optional[_t.Union[str, tuple, list]] = DEFAULT_FIT_LABEL,
        color: _t.Optional[_t.Union[str, int]] = None,
        title: _t.Optional[_t.Union[str, tuple, list]] = None,
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
            # label = format_str_with_params(self.res, label)
            if isinstance(label, (tuple, list)):
                label = "; ".join([str(ll) for ll in label])
            label = str(label).strip()
            kwargs.update({"label": label})

        color = get_right_color(color)
        kwargs.update({"color": color})

        if post_func_x:
            x_fit = post_func_x(x_fit)
        if post_func_y:
            y_fit = post_func_y(y_fit)
        ax.plot(x_fit, y_fit, **kwargs)

        if title:
            # title = format_str_with_params(self.res, title)
            if isinstance(title, (tuple, list)):
                title = "; ".join([str(t) for t in title])
            current_title = ax.get_title()
            if current_title:
                title = f"{current_title}\n{title}"
            title = str(title).strip()
            ax.set_title(title)

        if label != DEFAULT_FIT_LABEL and label is not None:
            ax.legend()

        return self

    # def plot_thick(
    #     self,
    #     ax: _t.Optional["Axes"] = None,
    #     *,
    #     x: _t.Optional[_t.Union[_NDARRAY, int]] = None,
    #     label: str = DEFAULT_FIT_LABEL,
    #     color: _t.Optional[_t.Union[str, int]] = None,
    #     title: _t.Optional[str] = None,
    #     kwargs_fill: _t.Optional[_t.Dict[str, _t.Any]] = None,
    #     **kwargs,
    # ):
    #     """Plot the fit results on the given axes.

    #     Args:
    #         ax (Optional[Axes]): The axes on which to plot the fit results. If None, a new axes will be created.
    #         label (str): The label for the plot. Defaults to ffit.config.DEFAULT_FIT_LABEL.
    #         color (Optional[Union[str, int]]): The color of the plot. If None, a default color will be used.
    #         title (Optional[str]): The title for the plot. If provided, it will be appended to the existing title.
    #         **kwargs: Additional keyword arguments to be passed to the plot function.

    #     Returns:
    #         FitResults: The FitResults object itself.

    #     Example:
    #         ```
    #         >>> result = ff.Cos().fit(x, y)
    #         >>> result.plot() # ax will be get from plt.gca()
    #         >>> result.plot(ax, x=x, label="Cosine fit")
    #         >>> result.plot(ax, x=x, label="Cosine fit", color="r")
    #         ```
    #     Worth to mention: title will be appended to the existing title with a new line.

    #     """
    #     ax = get_ax_from_gca()
    #     x_fit = get_right_x(x, ax, self.x)

    #     y_fit = self.res_func(x_fit)
    #     y_std = self.stdfunc(x_fit)

    #     y_1 = y_fit - y_std
    #     y_2 = y_fit + y_std

    #     label = format_str_with_params(self.res, label)

    #     color = get_right_color(color)
    #     # return x_fit, y_1, y_2

    #     # ax.plot(x_fit, y_fit, label=label, color=color, **kwargs)

    #     kwargs_fill = kwargs_fill or {}
    #     kwargs_fill.setdefault("color", color)
    #     ax.fill_between(x_fit, y_1, y_2, **kwargs_fill)  # type: ignore
    #     kwargs.setdefault("ls", "--")
    #     ax.plot(x_fit, np.mean([y_1, y_2], axis=0), label=label, color=color, **kwargs)

    #     if title:
    #         title = format_str_with_params(self.res, title)
    #         current_title = ax.get_title()
    #         if current_title:
    #             title = f"{current_title}\n{title}"
    #         ax.set_title(title)

    #     if label != DEFAULT_FIT_LABEL:
    #         ax.legend()

    #     return self
