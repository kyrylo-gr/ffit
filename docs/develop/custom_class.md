---
title: Custom Function Class
---

## Custom Function Class

This guide explains how to create a custom function class for use with the `ffit` package.

To contribute a new class, create a new file in the `ffit/funcs` directory. Name it `custom_func.py` and include two classes: `CustomFuncParam` and `CustomFunc`.

## Create a Param Class

Start by defining the function parameters in a class that inherits from `ParamDataclass`.

Also define the final result class that inherits from `FitResult` and `CustomFuncParam` to provide users with accurate typing.
The final result class should have attribute `param_class` set to the converted `CustomFuncParam` class.

```python
from dataclasses import dataclass
from ffit.utils import FuncParamClass, convert_param_class
from ffit.fit_results import FitResult

class CustomFuncParam(FuncParamClass):
    """CustomFunc function parameters.

    Attributes
    ----------
    - attr1: float
        First attribute of the function.
    - attr2: float
        Second attribute of the function.

    Additional attributes
    ----------------------
    - param1: float
        The param1 of the function.

    Methods
    -------
    - meth1: float
        The meth1 of the function.
    """
    __slots__ = ("attr1", "attr2")
    keys = ("attr1", "attr2")

    @property
    def param1(self):
        return self.attr1 ** 2 # pylint: disable=E1101

    def meth1(self):
        return self.attr1 * self.attr2 # pylint: disable=E1101

class CustomFuncResult(CustomFuncParam, FitResult[CustomFuncParam]):
    param_class = convert_param_class(CustomFuncParam)
```

## Define the Function

Define the function to be used for fitting.

The first argument is the x data (type `NDARRAY`), followed by the parameters in the same order as defined in `CustomFuncParam.keys`. The return type should be `NDARRAY`.

```python
from ffit.utils import _NDARRAY

def custom_func(x: _NDARRAY, attr1: float, attr2: float) -> _NDARRAY:
    return attr1 * attr2
```

## Define the Guess Function

The guess function helps determine initial parameters.

The first two arguments are the x and y data (type `NDARRAY`), followed by `**kwargs`. The return type should be `NDARRAY`.

You can pass `kwargs` to the `fit` methods to improve guesses. For example, you might specify whether an exponential function is increasing or decreasing.

```python
from ffit.utils import _NDARRAY
import numpy as np

def custom_func_guess(x: _NDARRAY, y: _NDARRAY, **kwargs) -> _NDARRAY:
    sign = kwargs.get("func_sign", np.sign(np.mean(y)))
    return np.array([...])
```

## Normalize the Result

Normalize results to avoid stochastic behavior in the fit function.

For example, ensure phases are $2\pi$ periodic or set one parameter to always be positive.

```python
from typing import Sequence
import numpy as np

def normalize_res_list(x: Sequence[float]) -> _NDARRAY:
    return np.array([
        abs(x[0]),
        np.sign(x[0]) * x[1],
        x[2] % (2 * np.pi),
        x[3]
    ])
```

## Define the Class

The main class should inherit from `FitLogic[CustomFuncResult]`. Set `CustomFuncResult` correctly to provide users with accurate typing.

Include a minimalistic docstring with LaTeX and Python representations of the function, as well as a reference to the final parameters class.

```python
from ffit.fit_logic import FitLogic
import typing as _t

class CustomFunc(FitLogic[CustomFuncResult]): # type: ignore
    r"""CustomFunc function.
    ---

    $$
    f(x)^2 = \cos(\omega)
    $$

        f(x) = sqrt(cos(2 * pi * frequency))

    Final Parameters
    -----------------
    The final parameters are given by [`CustomFuncParam`](../custom_func_param/) dataclass.
    """
    _result_class: _t.Type[CustomFuncResult] = CustomFuncResult

    func = staticmethod(custom_func)
    normalize_res = staticmethod(normalize_res_list)
    _guess = staticmethod(custom_func_guess)
```

## Additional Information

In order to submit a new function, each function class should include the following information.

### Example Parameters

Provide example parameters for documentation plots:

```python
    _example_param = (1, 1, 1.0, 1.0)
    # Additionally you can setup the x axis:
    _example_x_min: float
    _example_x_max: float
    _example_x_points: int
    _example_std: float
```

### Mask Signature

Add a mask method signature for better autocompletion.

Ensure the attributes match those in `CustomFuncParam`.

```python
    @overload
    @classmethod
    def mask(  # type: ignore # pylint: disable=W0221
        cls,
        *,
        attr1: float = None, # type: ignore
        attr2: float = None, # type: ignore
    ) -> "CustomFunc": ...

    @classmethod
    def mask(cls, **kwargs) -> "CustomFunc":
        return super().mask(**kwargs)
```

### Testing Range

Specify attribute testing ranges to ensure proper function behavior.

If not set, the default range is (-100, 100).

```python
    _test_range = {
        "attr1": (0, 1),
        "attr2": None,
    }
```
