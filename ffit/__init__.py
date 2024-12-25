"""FFit module
"""

# flake8: noqa
import typing as _t

from .__config__ import __version__
from .backends import SCIPY as DEFAULT_BACKEND
from .backends import BackendProtocol, Backends, get_backend
from .fit_logic import FitLogic, FitResult
from .front import async_curve_fit, async_curve_fit_array, curve_fit
from .funcs import *


def nest_asyncio_apply():
    import nest_asyncio

    nest_asyncio.apply()


CURRENT_BACKEND: _t.Optional[BackendProtocol] = None


def use_backend(backend: str):
    global CURRENT_BACKEND  # pylint: disable=W0603
    CURRENT_BACKEND = get_backend(backend)
