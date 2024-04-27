import inspect
import re
import typing as _t

import numpy as np

from .config import DEFAULT_PRECISION

# _NDARRAY = _t.Union[np.ndarray, jnp.ndarray]
# _ARRAY = _t.Union[_t.Sequence[jnp.ndarray], jnp.ndarray, np.ndarray]

_NDARRAY = np.ndarray
_ARRAY = _t.Union[_t.Sequence[np.ndarray], np.ndarray]


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


def param_len(cls):
    return len(cls.__annotations__)


_DEFAULT_COLORS: _t.Optional[_t.Dict[int, str]] = None


def get_color_by_int(index: int) -> _t.Optional[str]:
    global _DEFAULT_COLORS  # pylint: disable=W0603
    if _DEFAULT_COLORS is None:
        import matplotlib as mpl

        _DEFAULT_COLORS = dict(enumerate(mpl.rcParams["axes.prop_cycle"].by_key()["color"]))

    return _DEFAULT_COLORS.get(index % len(_DEFAULT_COLORS))


def get_right_color(color: _t.Optional[_t.Union[str, int]]) -> _t.Optional[str]:
    if isinstance(color, int) or (isinstance(color, str) and color.isdigit() and len(color) == 1):

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
        if parameters is None:
            return
        self._order = {name: i for i, (name, _) in enumerate(parameters)}

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args)


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
        (param.name, param.default if param.default is not inspect.Parameter.empty else None)
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
