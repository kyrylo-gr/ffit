# flake8: noqa: F401

import typing as _t

from ..fit_logic import FitLogic
from .complex_spiral import ComplexSpiral, ComplexSpiralParam
from .cos import Cos, CosParam
from .exp import Exp, ExpParam
from .gaussian import Gaussian, GaussianParam
from .hyperbola import Hyperbola, HyperbolaParam
from .line import Line, LineParam
from .log import Log, LogParam
from .lorentzian import Lorentzian, LorentzianParam
from .lorentzian_complex import LorentzComplex, LorentzParam

FIT_FUNCTIONS: _t.Dict[str, _t.Type[FitLogic]] = {
    "cos": Cos,
    "sin": Cos,
    "line": Line,
    "hyperbola": Hyperbola,
    "damped_exp": Exp,
    "complex_spiral": ComplexSpiral,
    "lorentz": LorentzComplex,
}
