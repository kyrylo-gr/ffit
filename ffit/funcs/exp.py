import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import _NDARRAY, FuncParamClass, check_min_len, convert_param_class

__all__ = ["Exp"]

_T = _t.TypeVar("_T")


class ExpParam(_t.Generic[_T], FuncParamClass):
    """Exponential function parameters.

    Attributes:
        amplitude (float):
            The amplitude of the exponential function.
        rate (float):
            The rate of the exponential function.
        offset (float):
            The offset of the exponential function.

    Additional attributes:
        tau (float):
            The time constant of the exponential function, calculated as -1 / rate.
    """

    keys = ("amplitude", "rate", "offset")

    amplitude: _T
    rate: _T
    offset: _T

    @property
    def tau(self) -> _T:
        return -1 / self.rate  # type: ignore # pylint: disable=E1101


class ExpResult(ExpParam, FitResult[ExpParam]):
    param_class = convert_param_class(ExpParam)


def exp_func(x, amplitude, rate, offset):
    return amplitude * np.exp(rate * x) + offset


def exp_error(x, amplitude, rate, offset, amplitude_std, rate_std, offset_std):
    """Calculate error due to the propagated uncertainty.

    Parameters:
        x (float): The input value.
        amplitude (float): The amplitude of the exponential function.
        rate (float): The rate of the exponential function.
        offset (float): The offset of the exponential function.
        amplitude_std (float): The standard deviation of the amplitude.
        rate_std (float): The standard deviation of the rate.
        offset_std (float): The standard deviation of the offset.

    Returns:
        float: The calculated error.
    """
    del offset
    # ∂f/∂A = e^(r*x)*ΔA
    due_amplitude = np.exp(rate * x) * amplitude_std
    # ∂f/∂r = A*x*e^(r*x)*Δr
    due_rate = amplitude * np.exp(rate * x) * x * rate_std
    # ∂f/∂O = 1
    due_offset = offset_std
    return np.sqrt(due_amplitude**2 + due_rate**2 + due_offset**2)


def exp_guess(x: _NDARRAY, y: _NDARRAY, **kwargs):
    """Provide an initial guess for the parameters of an exponential function.

    Parameters:
    x (_NDARRAY): The input data.
    y (_NDARRAY): The output data.
    **kwargs: Additional keyword arguments (ignored).

    Returns:
    _NDARRAY: An array containing the initial guess for the parameters of the exponential function.
    """
    del kwargs
    if not check_min_len(x, y, 3):
        return np.ones(3)
    average_size = max(len(y) // 10, 1)

    data = np.array([x, y]).T
    sorted_data = data[data[:, 0].argsort()]

    x1 = np.mean(sorted_data[:average_size, 0])
    y1 = np.mean(sorted_data[:average_size, 1])
    x3 = np.mean(sorted_data[-average_size:, 0])
    y3 = np.mean(sorted_data[-average_size:, 1])

    median = (x3 + x1) / 2
    median_index = np.abs(sorted_data[:, 0] - median).argmin()
    x2 = np.mean(sorted_data[median_index - average_size : median_index, 0])
    y2 = np.mean(sorted_data[median_index - average_size : median_index, 1])

    if x1 == x2 or x2 == x3 or x1 == x3:  # noqa
        return np.ones(3)
    if y1 == y2 or y2 == y3 or y1 == y3:  # noqa
        return np.ones(3)
    #  y1 = a * exp(b * x1) + c
    #  y2 = a * exp(b * x2) + c
    #  y3 = a * exp(b * x3) + c
    # y1 - y2 = a * (exp(b * x1) - exp(b * x2))
    # y3 - y2 = a * (exp(b * x3) - exp(b * x2))
    # (y1 - y2) / (y3 - y2) = (exp(b * x1) - exp(b * x2)) / (exp(b * x3) - exp(b * x2))
    # exp(b*x) ≈ 1 + b*x
    # (y1-y2)/(y3-y2) =
    # concave = 1 if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0 else -1
    # b = concave * (y2 - y1) / (y3 - y2) * (x3 - x2) / (x2 - x1) / 3

    # More simpler: tau = total_x / 3
    concave = 1 if (y3 - y1) / (x3 - x1) > (y2 - y1) / (x2 - x1) else -1
    b = concave * 3 / (x3 - x1) * np.sign(y3 - y1)

    z1 = np.exp(b * x1)
    z3 = np.exp(b * x3)
    #  y1 = a * exp(b * x1) + c = a * z1 + c
    #  y3 = a * exp(b * x3) + c = a * z3 + c
    # => a = (y1 - y3) / (z1 - z3)
    # => c = (y1 * z3 - y3 * z1) / (z3 - z1)
    a = (y1 - y3) / (z1 - z3)
    c = (y1 * z3 - y3 * z1) / (z3 - z1)
    return np.array([a, b, c])
    # offset = y1 if concave > 0 else y3
    # amplitude = concave * (y3 - y1) * np.sign(x3 - x1)

    # print(amplitude, concave)
    # print(amplitude, concave / abs(x3 - x1), offset)
    # print((x1, y1), (x2, y2), (x3, y3))
    # return amplitude, concave / abs(x3 - x1), offset  # - amplitude * np.exp(x1)


class Exp(FitLogic[ExpResult]):  # type: ignore
    r"""Exp function.


    Function
    ---------

    $$
        f(x) = A \exp(Γ⋅x) + A_{\text{offset}}
    $$

        f(x) = amplitude * np.exp(rate * x) + offset


    Final parameters
    -----------------
    The final parameters are given by [`ExpParam`](../exp_param/) dataclass.

    """

    _result_class: _t.Type[ExpResult] = ExpResult

    func = staticmethod(exp_func)
    func_std = staticmethod(exp_error)
    _guess = staticmethod(exp_guess)

    _example_param = (-3, -0.5, 3)
    _example_x_min = 0
    _example_x_max = 10

    @_t.overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        amplitude: float = None,  # type: ignore
        rate: float = None,  # type: ignore
        offset: float = None,  # type: ignore
    ) -> "Exp": ...

    @classmethod
    def mask(cls, **kwargs) -> "Exp":
        return super().mask(**kwargs)
