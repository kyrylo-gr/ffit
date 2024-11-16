import importlib
import pkgutil
from typing import Type

import numpy as np

import ffit
from ffit.fit_logic import FitLogic
from ffit.utils import ParamDataclass


def create_name(cls: Type[FitLogic], param, range_x=None):
    msg = f"{cls.__name__}({', '.join(map(str, param))})"
    if range_x is not None:
        msg += f" in {range_x}"
    return msg


def compare_result(cls, msg, res, param):
    assert res is not None, f"\n{msg}.\n Expected result, got None"
    assert len(res) == len(
        param
    ), f"\n{msg}.\n Expected {len(param)} parameters, got {len(res)}"
    for i, (r, p) in enumerate(zip(res, param)):
        assert np.isclose(
            r, p, rtol=getattr(cls, "_test_rtol", 1e-3)
        ), f"\n{msg}.\n At {i} expected {p}, got {r}"


def run_fit(cls: Type[FitLogic], param, range_x):
    min_x, max_x = range_x
    x = np.linspace(min_x, max_x, 1000)
    y = cls.func(x, *param)

    res = cls().fit(x, y).res
    compare_result(cls, f"fit:{create_name(cls, param, (min_x, max_x))}", res, param)


def run_mask_fit(cls: Type[FitLogic], param, range_x):
    min_x, max_x = range_x
    x = np.linspace(min_x, max_x, 1000)
    param[0] = 1
    y = cls.func(x, *param)

    res = cls.mask(**{cls.param.fields()[0]: 1}).fit(x, y).res
    compare_result(
        cls, f"mask_fit:{create_name(cls, param, (min_x, max_x))}", res, param
    )


def generate_params(cls: Type[FitLogic]):
    param_cls: ParamDataclass = cls.param  # type: ignore
    param_len = len(param_cls.fields())
    range_x = getattr(cls, "_range_x", (-np.inf, np.inf))
    range_x = max(range_x[0], -10), min(range_x[1], 10)

    yield [1] * param_len, range_x
    # yield (np.arange(param_len) + 1) / param_len, range_x


def run_cls(cls: Type[FitLogic]):
    for param, range_x in generate_params(cls):
        run_fit(cls, param, range_x)
        run_mask_fit(cls, param, range_x)


def test_all_functions():
    for _, module_name, _ in pkgutil.iter_modules(ffit.funcs.__path__):
        module = importlib.import_module(f"ffit.funcs.{module_name}")
        for _, cls in list(vars(module).items()):
            if (
                isinstance(cls, type)
                and issubclass(cls, FitLogic)
                and cls is not FitLogic
            ):
                if getattr(cls, "_test_ignore", False):
                    continue
                run_cls(cls)
