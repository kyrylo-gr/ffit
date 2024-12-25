---
title: Log
---

### Simples example

```
>>> from ffit.funcs.log import Log

# Call the fit method with x and y data.
>>> fit_result = Log().fit(x, y)

# The result is a FitResult object that can be unpacked.
>>> res, res_func = fit_result.res_and_func()

# The parameters can be accessed as attributes.
>> amplitude = fit_result.amplitude

# One can combine multiple calls in one line.
>>> res = Log().fit(x, y, guess=[1, 2, 3, 4]).plot(ax)
```

### Final parameters

<!-- prettier-ignore -->
::: ffit.funcs.log.LogParam
    options:
      show_bases: false
      show_root_heading: false
      summary: false


<!-- prettier-ignore -->
::: ffit.funcs.log.Log


