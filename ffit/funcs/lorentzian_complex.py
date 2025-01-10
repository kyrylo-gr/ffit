import typing as _t

import numpy as np

from ..fit_logic import FitLogic
from ..fit_results import FitResult
from ..utils import FuncParamClass, convert_param_class

__all__ = ["LorentzComplex"]

_T = _t.TypeVar("_T")


class LorentzParam(_t.Generic[_T], FuncParamClass):
    """General Lorentz parameters.

    Attributes:
        a (float)
        b (float)
        b0 (float)
        c (float)
        d (float)
        d0 (float)
        r (float)
        amplitude0 (float)
        amplitude_phase (float)
    """

    keys = ("a", "b", "b0", "c", "d", "d0", "r", "amplitude0", "amplitude_phase")

    a: _T
    b: _T
    b0: _T
    c: _T
    d: _T
    d0: _T
    r: _T
    amplitude0: _T
    amplitude_phase: _T


class LorentzResult(LorentzParam, FitResult[LorentzParam]):
    param_class = convert_param_class(LorentzParam)


def lorentz_func(x, a, b, b0, c, d, d0, r, amplitude0, amplitude_phase):
    amplitude = amplitude0 * np.exp(1j * amplitude_phase)
    return amplitude * (a + b * (x - b0)) / (c + d * (x - d0)) * np.exp(1j * x * r)


def lorentz_guess(x, y, **kwargs):
    del x, y, kwargs
    return np.array([1, 1, 0, 1, 1, 0, 0, 1, 0])


def lorentz_transmission():
    pass


def lorentz_reflection():
    pass


class LorentzTransmission(FitLogic[LorentzResult]):  # type: ignore
    _doc_ignore = True
    _test_ignore = True


class LorentzReflection(FitLogic[LorentzResult]):  # type: ignore
    _doc_ignore = True
    _test_ignore = True


class LorentzComplex(FitLogic[LorentzResult]):  # type: ignore
    r"""Lorentz Transmission function.
    ---------
    General Lorentzian function can be written as:
    $$
    f(x) = Z_0 e^{i⋅x⋅r} \frac{a + b⋅(x-b_0)}{c + d⋅(x-d_0)}
    $$

        f(x) = (
            amplitude0 * np.exp(1j * amplitude_phase)
            *  (a + b * (x - b0)) / (c + d * (x - d0))
            * np.exp(1j * x * r)
        )


    Final parameters
    -----------------
    The final parameters are given by [`LorentzianParam`](../lorentzian_param/) dataclass.


    """

    _result_class: _t.Type[LorentzResult] = LorentzResult

    func = staticmethod(lorentz_func)
    _guess = staticmethod(lorentz_guess)  # type: ignore

    _test_ignore = True
    _doc_ignore = True

    Transmission = LorentzTransmission
    Reflection = LorentzReflection

    # TODO: def correct_phase(x, z:)
