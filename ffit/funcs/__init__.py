# flake8: noqa: F401

import typing as _t

from ..fit_logic import FitLogic
from .complex_spiral import ComplexSpiral
from .cos import Cos
from .exp import Exp
from .gaussian import Gaussian
from .hyperbola import Hyperbola
from .line import Line
from .log import Log
from .lorentzian import Lorentzian
from .lorentzian_complex import LorentzComplex

FIT_FUNCTIONS: _t.Dict[str, _t.Type[FitLogic]] = {
    "cos": Cos,
    "sin": Cos,
    "line": Line,
    "hyperbola": Hyperbola,
    "damped_exp": Exp,
    "complex_spiral": ComplexSpiral,
    "lorentz": LorentzComplex,
    "gaussian": Gaussian,
    "log": Log,
    "lorentzian": Lorentzian,
}
