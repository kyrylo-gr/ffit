from .protocol import BackendABC

# https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf


class JAXBackend(BackendABC):
    @classmethod
    def curve_fit(cls, func, xdata, ydata, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def leastsq(cls, func, p0, *args, **kwargs):
        raise NotImplementedError
