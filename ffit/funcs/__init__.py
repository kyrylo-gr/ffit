# flake8: noqa: F401
import typing as _t

from ..backends import SCIPY as DEFAULT_BACKEND
from ..backends import BackendProtocol, Backends, get_backend
from .complex_spiral import ComplexSpiral, ComplexSpiralParam
from .cos import Cos, CosParam
from .damped_exp import DampedExp, DampedExpParam
from .hyperbola import Hyperbola, HyperbolaParam
from .line import Line, LineParam
from .lorentz import LorentzComplex, LorentzParam

CURRENT_BACKEND: _t.Optional[BackendProtocol] = None


def use_backend(backend: str):
    global CURRENT_BACKEND  # pylint: disable=W0603
    CURRENT_BACKEND = get_backend(backend)
