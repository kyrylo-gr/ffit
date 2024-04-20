from scipy import optimize  # curve_fit, leastsq

from .protocol import BackendABC, RunResult


class ScipyBackend(BackendABC):
    @classmethod
    def curve_fit(cls, func, xdata, ydata, *args, **kwargs):
        popt, pcov = optimize.curve_fit(func, xdata, ydata, *args, **kwargs)

        return RunResult(popt, {"pcov": pcov})

    @classmethod
    def leastsq(cls, func, p0, *args, **kwargs):
        popt, pcov = optimize.leastsq(func, p0, *args, **kwargs)

        return RunResult(popt, {"pcov": pcov})
