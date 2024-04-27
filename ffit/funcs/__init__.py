# flake8: noqa: F401

import typing as _t

from ..fit_logic import FitLogic
from .complex_spiral import ComplexSpiral, ComplexSpiralParam
from .func_cos import Cos, CosParam
from .func_exp import Exp, ExpParam
from .func_hyperbola import Hyperbola, HyperbolaParam
from .func_line import Line, LineParam
from .func_log import Log, LogParam
from .func_lorentzian import Lorentzian, LorentzianParam
from .func_lorentzian_complex import LorentzComplex, LorentzParam

FIT_FUNCTIONS: _t.Dict[str, _t.Type[FitLogic]] = {
    "cos": Cos,
    "sin": Cos,
    "line": Line,
    "hyperbola": Hyperbola,
    "damped_exp": Exp,
    "complex_spiral": ComplexSpiral,
    "lorentz": LorentzComplex,
}
