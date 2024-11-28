import abc
import typing as _t

import numpy as np


class RunResult(_t.NamedTuple):
    params: np.ndarray
    info: _t.Dict[str, _t.Any]


class BackendABC(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def curve_fit(
        cls,
        func,
        xdata,
        ydata,
        p0=None,
        sigma=None,
        absolute_sigma=False,
        check_finite=None,
        bounds=(-np.inf, np.inf),
        method=None,
        jac=None,
        *,
        full_output=False,
        nan_policy=None,
        **kwargs
    ) -> RunResult:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def leastsq(
        cls,
        func,
        p0,
        args=(),
        Dfun=None,
        full_output=False,
        col_deriv=False,
        ftol: float = 1.49012e-08,
        xtol: float = 1.49012e-08,
        gtol: float = 0.0,
        maxfev: int = 0,
        epsfcn=None,
        factor: int = 100,
        diag=None,
    ) -> RunResult:
        raise NotImplementedError


class BackendProtocol(_t.Protocol):
    @classmethod
    def curve_fit(cls, *args, **kwargs) -> RunResult: ...  # noqa: E704

    @classmethod
    def leastsq(cls, *args, **kwargs) -> RunResult: ...  # noqa: E704
