---
title: Lorentzian
---

### Simples example

```
>>> from ffit.funcs.lorentzian import Lorentzian

# Call the fit method with x and y data.
>>> fit_result = Lorentzian().fit(x, y)

# The result is a FitResult object that can be unpacked.
>>> res, res_func = fit_result.res_and_func()

# The parameters can be accessed as attributes.
>> amplitude = fit_result.amplitude

# One can combine multiple calls in one line.
>>> res = Lorentzian().fit(x, y, guess=[1, 2, 3, 4]).plot(ax)
```

### Final parameters

<!-- prettier-ignore -->
::: ffit.funcs.lorentzian.LorentzianParam
    options:
      show_bases: false
      show_root_heading: false
      summary: false


<!-- prettier-ignore -->
::: ffit.funcs.lorentzian.Lorentzian


