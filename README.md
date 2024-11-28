# FFit. Fit python library.

<h1 align="center">
<img src="./docs/images/ffit-logo.png" width="400">
</h1><br>

[![Pypi](https://img.shields.io/pypi/v/ffit.svg)](https://pypi.org/project/ffit/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License](https://img.shields.io/badge/license-LGPL-green)](./LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CodeFactor](https://www.codefactor.io/repository/github/kyrylo-gr/ffit/badge/main)](https://www.codefactor.io/repository/github/kyrylo-gr/ffit/overview/main)
[![Codecov](https://codecov.io/gh/kyrylo-gr/ffit/graph/badge.svg?token=5U0FU9XNID)](https://codecov.io/gh/kyrylo-gr/ffit)
[![Download Stats](https://img.shields.io/pypi/dm/ffit)](https://pypistats.org/packages/ffit)
[![Documentation](https://img.shields.io/badge/docs-blue)](https://kyrylo-gr.github.io/ffit/)

`FFit` - Python library for easier fitting.

## Install

`pip install ffit`

For more installation details, please refer to the [How to install](starting_guide/install.md)

## How to use

The aim of this library is to simplify the fit routine. Here are some examples of how to use it.

### Simple syntax

```
import ffit as ff

x = np.linspace(1, 10, 100)
y = 2 * np.sin(x) + 3

res = ff.Cos().fit(x, y).res

```

### Plotting result

```
import ffit as ff
import matplotlib.pyplot as plt

x = np.linspace(1, 10, 100)
y = 2 * np.sin(x) + 3

plt.plot(x, y, '.')

res = ff.Cos().fit(x, y).plot().res
```

### Plotting guess

The quality of fitting is heavily dependent on the initial guess. This library provides initial guesses for various popular functions to ensure effectiveness. However, if something goes awry, you can verify the guess and set it manually.

```
ff.Cos().guess(x, y).plot()

ff.Cos().guess(x, y, guess=[1,2,3,4]).plot()

ff.Cos().fit(x, y, guess=[1,2,3,4]).plot()
```

### Other functions

Numerous functions are available out of the box. You can refer to [the documentation](https://kyrylo-gr.github.io/ffit/functions/) for more details.

Moreover, you can use your custom functions with familiar syntax and still benefit from existing routines.

## Any custom function

For any custom function, you can create a custom class for a guessing algorithm or use `curve_fit` or `least_square` like this:

```
def line_func(x, amp, offset):
    return amp * x + offset

fit_amp = ff.curve_fit(line_func, x, y).plot().res.amp

```
