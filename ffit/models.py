from typing import Any, Callable, List, NamedTuple, Optional, Protocol, Union

import numpy as np
import scipy.optimize as optimize
from . import fit_utils as utils


class FitModel:
    """Fit model"""

    def __init__(self, func: Callable, params: Any) -> None:
        """Initial the FitModel with fitting function and params of class . Function is provided fuction for fitting"""
        self.func = func
        self.params = params

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict function on values"""
        return self.func(x, *self.params)

    def error(self) -> float:
        """Error function"""
        return 0


class FitterFunction(Protocol):
    """Fit function"""
    @staticmethod
    def guess(x: np.ndarray, y: np.ndarray) -> NamedTuple:
        ...

    @staticmethod
    def func(x: np.ndarray, *args) -> np.ndarray:
        ...


class Fitter:
    """Fit

    Examples
    --------
    ::

        f = Fitter(Polynomial(3))
        fm = f.sfit(x, y_data)
        y_pred = fm.predict(x)
        plt.plot(x, y_data, '.')
        plt.plot(x, y_pred)

    """

    def __init__(self, func_to_fit: FitterFunction):
        self.func_to_fit = func_to_fit
        self.x0 = None

    def guess(self, x: np.ndarray, y: np.ndarray):
        return self.func_to_fit.guess(x, y)

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            x0: Optional[Union[List, NamedTuple]] = None
            ) -> FitModel:

        x0 = self.guess(x, y) if x0 is None else x0

        fit = optimize.minimize(
            fun=lambda params: utils.chisq(self.func_to_fit.func, x, y, params),
            x0=x0,
            jac="3-point",
            method="SLSQP"
        ).x

        return FitModel(self.func_to_fit.func, fit)

    def sfit(self,
             x: np.ndarray,
             y: np.ndarray,
             x0: Optional[Union[List, NamedTuple]] = None,
             temp: int = 1
             ) -> FitModel:

        x0 = self.func_to_fit.guess(x, y) if x0 is None else x0
        x0_list = list(x0)

        fit = optimize.basinhopping(
            func= lambda params: utils.chisq(self.func_to_fit.func, x, y, params)),
            x0 = list(x0_list),
            T = temp,
            # minimizer_kwargs={"jac": lambda params: chisq_jac(sin_jac, x, y_data, params)}
        ).x

        return FitModel(self.func_to_fit.func, fit)
