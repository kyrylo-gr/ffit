import re
import typing as _t

import jax.numpy as jnp
import numpy as np

_ARRAY = _t.Union[_t.Sequence[jnp.ndarray], jnp.ndarray, np.ndarray]
_NDARRAY = _t.Union[np.ndarray, jnp.ndarray]


def get_mask(
    mask: _t.Optional[_t.Union[_ARRAY, float, list]] = None,
    x: _t.Optional[_ARRAY] = None,
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
        return jnp.ones_like(jnp.array(x), dtype=bool)
    elif isinstance(mask, (int, float)):
        if x is None:
            raise ValueError("Mask cannot be float if x is not provided.")
        return jnp.array(x) < mask
    return jnp.array(mask)


def param_len(cls):
    return len(cls.__annotations__)


_DEFAULT_COLORS: _t.Optional[_t.Dict[int, str]] = None
DEFAULT_FIT_LABEL = "Fit"
DEFAULT_PRECISION = ".2f"


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
