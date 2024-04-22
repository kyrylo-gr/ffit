import typing as _t

import jax.numpy as jnp

from .utils import _NDARRAY, DEFAULT_FIT_LABEL, format_str_with_params, get_right_color

_R = _t.TypeVar("_R", bound=_t.Sequence)
if _t.TYPE_CHECKING:
    from matplotlib.axes import Axes


def get_ax_from_gca() -> "Axes":
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
        x = line.get_xdata()
        if expected_len and len(x) != expected_len:
            raise ValueError("X must be provided. Cannot be extracted from the plot.")
        return jnp.array(x)
    raise ValueError("X must be provided.")


def create_x_from_ax(ax: "Axes", x: _t.Optional[_NDARRAY] = None) -> _NDARRAY:
    if x is None:
        lims = ax.get_xlim()
        return jnp.linspace(*lims, 200)
    if len(x) < 100:
        return jnp.linspace(jnp.min(x), jnp.max(x), 200)
    return x


class FitResult(_t.Tuple[_t.Optional[_R], _t.Callable]):
    """
    Represents the result of a fit operation.

    Attributes:
        res: The result of the fit operation.
        res_func: A callable function that takes a numpy array as input and returns a numpy array as output.
    """

    res: _t.Optional[_R]
    res_func: _t.Callable[[_NDARRAY], _NDARRAY]
    x: _t.Optional[_NDARRAY]

    def __init__(
        self,
        res: _t.Optional[_R] = None,
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
        self.res_func = res_func if res_func is not None else (lambda x: jnp.ones_like(x) * jnp.nan)
        self.x = x

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
        label: str = DEFAULT_FIT_LABEL,
        color: _t.Optional[_t.Union[str, int]] = None,
        title: _t.Optional[str] = None,
        **kwargs,
    ):
        if ax is None:
            ax = get_ax_from_gca()

        x_fit = create_x_from_ax(ax, self.x)
        y_fit = self.res_func(x_fit)

        label = format_str_with_params(self.res, label)

        color = get_right_color(color)

        ax.plot(x_fit, y_fit, label=label, color=color, **kwargs)

        if title:
            title = format_str_with_params(self.res, title)
            current_title = ax.get_title()
            if current_title:
                title = f"{current_title}\n{title}"
            ax.set_title(title)

        if label != DEFAULT_FIT_LABEL:
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
        self.res_func = res_func if res_func is not None else (lambda x: jnp.ones_like(x) * jnp.nan)
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
            res_func = lambda x: jnp.ones_like(x) * jnp.nan  # noqa: E731

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
                return jnp.nan
            if isinstance(parameter, int):
                return res.res[parameter]  # type: ignore
            return getattr(res.res, parameter)

        return jnp.array([get_key(f) for f in self.res])

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
