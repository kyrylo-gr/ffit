# flake8: noqa: F401

from typing import Dict as _DICT

from ..fit_logic import FitLogic
from .complex_spiral import ComplexSpiral, ComplexSpiralParam
from .damped_exp import DampedExp, DampedExpParam
from .func_cos import Cos, CosParam
from .hyperbola import Hyperbola, HyperbolaParam
from .line import Line, LineParam
from .lorentz import LorentzComplex, LorentzParam

FIT_FUNCTIONS: _DICT[str, FitLogic] = {
    "cos": Cos,
    "sin": Cos,
    "line": Line,
    "hyperbola": Hyperbola,
    "damped_exp": DampedExp,
    "complex_spiral": ComplexSpiral,
    "lorentz": LorentzComplex,
}
