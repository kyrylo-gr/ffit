import inspect
import re
import typing as _t

import numpy as np

from .config import DEFAULT_PRECISION, DEFAULT_S_PRECISION

# _NDARRAY = _t.Union[np.ndarray, jnp.ndarray]
# _ARRAY = _t.Union[_t.Sequence[jnp.ndarray], jnp.ndarray, np.ndarray]

_NDARRAY = np.ndarray
_ARRAY = _t.Union[_t.Sequence[np.ndarray], np.ndarray, _t.Sequence[float]]
_2DARRAY = _t.Union[
    _t.Sequence[np.ndarray], np.ndarray, _t.Sequence[_t.Sequence[float]]
]
_T = _t.TypeVar("_T", bound=_t.Type)


def get_mask(
    mask: _t.Optional[_t.Union[_ARRAY, float, list]] = None,
    x: _t.Optional[_ARRAY] = None,
) -> np.ndarray:
    """Return a mask array based on the provided mask or threshold.

    Parameters:
    - mask: The mask array or threshold (optional).
    - x: The independent variable (optional).

    Returns:
    - np.ndarray: The mask array.

    """
    if mask is None:
        if x is None:
            raise ValueError("Either x or mask must be provided.")
        return np.ones_like(np.array(x), dtype=bool)
    elif isinstance(mask, (int, float)):
        if x is None:
            raise ValueError("Mask cannot be float if x is not provided.")
        return np.array(x) < mask
    return np.array(mask)


def get_masked_data(
    x: _NDARRAY,
    data: _NDARRAY,
    mask: _t.Optional[_t.Union[_ARRAY, float]],
    mask_min_len: int = 1,
) -> _t.Tuple[_NDARRAY, _NDARRAY]:
    mask = get_mask(mask, x)
    if np.sum(mask) < mask_min_len:
        return np.array([]), np.array([])
    return x[mask], data[mask]


def param_len(cls):
    return len(cls.__annotations__)


_DEFAULT_COLORS: _t.Optional[_t.Dict[int, str]] = None


def get_color_by_int(index: int) -> _t.Optional[str]:
    global _DEFAULT_COLORS  # pylint: disable=W0603
    if _DEFAULT_COLORS is None:
        import matplotlib as mpl

        _DEFAULT_COLORS = dict(
            enumerate(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        )

    return _DEFAULT_COLORS.get(index % len(_DEFAULT_COLORS))


def get_right_color(color: _t.Optional[_t.Union[str, int]]) -> _t.Optional[str]:
    if isinstance(color, int) or (
        isinstance(color, str) and color.isdigit() and len(color) == 1
    ):

        return get_color_by_int(int(color))
    return color


def format_str_with_params(
    params: _t.Optional[_t.Sequence[str]],
    text: str,
    default_precision: str = DEFAULT_PRECISION,
):
    if params is None or "$" not in text:
        return text

    possible_params = re.findall(r"\$(\d)(\.\d[fed])?", text)
    if not possible_params:
        return text
    for index, precision in possible_params:
        index = int(index)
        if index is None or index >= len(params):  # type: ignore
            continue
        if precision is None:
            precision = default_precision
            to_replace = f"${index}"
        else:
            to_replace = f"${index}{precision}"

        param = params[index]  # type: ignore
        text = text.replace(to_replace, f"{format(param, precision)}")

    return text


class DynamicNamedTuple(tuple):
    """
    A subclass of tuple that allows accessing elements by attribute name.

    This class provides a way to access elements of a tuple using attribute names instead of indices.
    It inherits from the built-in tuple class and overrides the __getattr__ method to enable attribute-based access.

    Attributes:
        _order (Dict[str, int]): A dictionary that maps attribute names to their corresponding indices in the tuple.
    """

    _order: _t.Dict[str, int]

    def __getattr__(self, name):
        """
        Get the element of the tuple based on the attribute name.

        Args:
            name (str): The attribute name.

        Returns:
            Any: The element of the tuple corresponding to the attribute name.

        Raises:
            AttributeError: If the attribute name is not found in the _order dictionary.
        """
        number = self._order[name]
        return self[number]

    def __init__(
        self,
        *args,
        parameters: _t.Optional[_t.List[_t.Tuple[str, _t.Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a DynamicNamedTuple object.

        Args:
            *args: Positional arguments passed to the tuple constructor.
            parameters (Optional[List[Tuple[str, Any]]]): A list of tuples representing the
                attribute names and their initial values.
            **kwargs: Keyword arguments passed to the tuple constructor.
        """
        del args, kwargs
        if parameters is None:
            return
        self._order = {name: i for i, (name, _) in enumerate(parameters)}

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args)

    def asdict(self) -> _t.Dict[str, _t.Any]:
        """
        Convert the DynamicNamedTuple to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the DynamicNamedTuple.
        """
        return {name: self[number] for name, number in self._order.items()}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.asdict()})"


def get_function_args_ordered(func: _t.Callable) -> _t.List[_t.Tuple[str, _t.Any]]:
    """For given function return the names of the arguments and their default values.

    Args:
        func (Callable): The function to inspect.

    Returns:
        List[Tuple[str, Any]]: An ordered list of tuples, where each tuple contains the argument name followed by its
        default value or None if the argument has no default value.
    """
    sig = inspect.signature(func)
    args_ordered = [
        (
            param.name,
            param.default if param.default is not inspect.Parameter.empty else None,
        )
        for param in sig.parameters.values()
    ]
    return args_ordered


def create_named_tuple(func: _t.Callable, data: _t.Sequence) -> DynamicNamedTuple:
    """Create a named tuple from a function and a sequence of data."""
    args_ordered = get_function_args_ordered(func)[1:]
    return DynamicNamedTuple(data, parameters=args_ordered)


def check_min_len(x: _t.Optional[_ARRAY], y: _t.Optional[_ARRAY], min_len: int) -> bool:
    if x is None or y is None:
        return False
    if len(x) < min_len or len(y) < min_len:
        return False

    return True


def get_random_subarrays(x, y, num_of_permutations: _t.Optional[int] = None):
    total_elements = len(x)
    if num_of_permutations is None:
        num_of_permutations = int(min(max(total_elements / 10, 1_000), 5_000))
    sub_y = []
    # num_elements_to_select = int(total_elements * selection_ratio)

    for _ in range(num_of_permutations):
        selected_indexes = np.random.choice(
            total_elements, size=total_elements, replace=True
        )

        # Create the subarray using the selected indexes
        sub_y.append([x[selected_indexes], y[selected_indexes]])

    return np.array(sub_y)


def bootstrap_generator(N, K):
    for _ in range(K):
        yield np.random.choice(N, N, replace=True)


class FuncParamProtocol(_t.Protocol):
    keys: _t.Tuple[str, ...]

    @classmethod
    def __len__(cls) -> int: ...


class FuncParamMeta(type):
    keys: _t.Tuple[str, ...] = tuple()

    def __len__(cls) -> int:
        return len(cls.keys)


class FuncParamClass(metaclass=FuncParamMeta):
    def asdict(self) -> _t.Dict[str, _t.Any]:
        return self._asdict()  # type: ignore # pylint: disable=E1101

    def __repr__(self):
        return f"{self.__class__.__name__}({self.asdict()})"


def mask_func(func, mask, mask_values):
    def masked_func(x, *args):
        if len(mask) == len(args):
            args = np.array(args)[mask]
        params_full = np.zeros_like(mask).astype(float)
        params_full[mask] = args
        params_full[~mask] = mask_values[~mask]
        return func(x, *params_full)

    return masked_func


def mask_func_result(func, mask):
    def masked_func(*args, **kwargs2):
        res = func(*args, **kwargs2)
        return res[mask]

    return masked_func


def std_monte_carlo(
    x: _NDARRAY,
    func: _t.Callable,
    means: _ARRAY,
    stds: _ARRAY,
    n_simulations: int = 10_000,
) -> _NDARRAY:
    # Arrays to hold the results of each simulation
    simulated_functions = np.zeros((n_simulations, len(x)))
    # Sampling from normal distribution
    values = np.array(
        [np.random.normal(m, s, n_simulations) for m, s in zip(means, stds)]
    )
    # Monte Carlo simulation
    for i in range(n_simulations):
        simulated_functions[i, :] = func(x, *values[:, i])

    return np.std(simulated_functions, axis=0)


class classproperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


def convert_param_class(cls: _T) -> _T:
    class ConvertedParamClass(cls):
        _pure_param_class = cls
        _array: _NDARRAY

        def __init__(self, *args, **kwargs):
            for key, arg in zip(self.keys, args):
                setattr(self, key, arg)
            for key, arg in kwargs.items():
                setattr(self, key, arg)

            self._array = np.array([getattr(self, key) for key in self.keys])

        def __repr__(self):
            return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self, key)}' for key in self.keys])})"

        def __iter__(self):
            return iter(self._array)

        def __len__(self):
            return len(self._array)

    return ConvertedParamClass


class EquationClass:
    name: str

    def __init__(self, name, val, units=""):
        self.name = name
        self.val = val
        self.units = units

    @property
    def real(self) -> "EquationClass":
        """Get the real part."""
        return EquationClass(f"Re({self.name})", self.val.real, self.units)

    @property
    def imag(self) -> "EquationClass":
        """Get the imaginary part."""
        return EquationClass(f"Im({self.name})", self.val.imag, self.units)

    @property
    def abs(self) -> "EquationClass":
        """Get the absolute value."""
        return EquationClass(f"|{self.name}|", abs(self.val), self.units)

    @property
    def angle(self) -> "EquationClass":
        """Get the angle in radians."""
        return EquationClass(f"Arg({self.name})", np.angle(self.val), self.units)

    @property
    def deg(self) -> "EquationClass":
        """Get the angle in degrees."""
        return EquationClass(f"{self.name}°", np.angle(self.val, deg=True), self.units)

    def f(self, format_spec: _t.Optional[str] = None) -> str:
        """Format the value of the equation and return it as a string.

        If format_spec is provided, it will be used to format the value.
        IF format_spec is not provided, the value will be formatted automatically
        to DEFAULT_PRECISION or DEFAULT_S_PRECISION.
        """
        if format_spec is not None:
            return f"{self: {format_spec}}"
        return self._auto_format()

    @property
    def s(self) -> str:
        """Autoformat the value of the equation and return it as a string."""
        return self._auto_format()

    def n(self, new_name) -> "EquationClass":
        """Change the name of the equation."""
        return EquationClass(new_name, self.val, self.units)

    def l(self, new_name) -> "EquationClass":  # noqa: E743
        """Change the name of the equation."""
        return EquationClass(new_name, self.val, self.units)

    def u(
        self, new_units: str, coef: _t.Optional[_t.Union[float, int]] = None
    ) -> "EquationClass":
        """Change the units of the equation and multiply the value by a coefficient.
        If the coefficient is an integer, it will be used as a power of 10.
        """
        if isinstance(coef, int):
            coef = 10**coef
        elif coef is None:
            coef = 1
        return EquationClass(self.name, self.val * coef, new_units)

    def __format__(self, format_spec):
        val = f"{self.val:{format_spec}}"
        return f"{self.name} = {val}{self._get_units()}"

    def __str__(self):
        return self._auto_format()

    def __repr__(self):
        return f"{self.name} = {self.val}{self._get_units()}"

    def _get_units(self) -> str:
        if self.units:
            return f" {self.units}"
        return ""

    def _auto_format(self) -> str:
        if self.val < 2e3:
            val = f"{self.val: {DEFAULT_PRECISION}}"
            if val.strip(" -0.") == "":
                val = f"{self.val:{DEFAULT_S_PRECISION}}"
        else:
            val = f"{self.val:{DEFAULT_S_PRECISION}}"

        val = val.replace("e+0", "e").replace("e-0", "e-")

        return f"{self.name} = {val}{self._get_units()}"


class LabelClass:
    def __init__(self, params):
        self._params = params

    def __getattr__(self, name: str):
        val = getattr(self._params, name)
        if isinstance(val, (int, float, complex)):
            return EquationClass(name, val)
        return val


def convert_to_label_instance(param_class, array):

    return LabelClass(param_class(*array))
