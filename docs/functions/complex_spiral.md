---
title: ComplexSpiral
---

### Simples example

```
>>> from ffit.funcs.complex_spiral import ComplexSpiral

# Call the fit method with x and y data.
>>> fit_result = ComplexSpiral().fit(x, y)

# The result is a FitResult object that can be unpacked.
>>> res, res_func = fit_result.res_and_func()

# The parameters can be accessed as attributes.
>> amplitude0 = fit_result.amplitude0

# One can combine multiple calls in one line.
>>> res = ComplexSpiral().fit(x, y, guess=[1, 2, 3, 4]).plot(ax)
```

### Final parameters

<!-- prettier-ignore -->
::: ffit.funcs.complex_spiral.ComplexSpiralParam
    options:
      show_bases: false
      show_root_heading: false
      summary: false


<!-- prettier-ignore -->
::: ffit.funcs.complex_spiral.ComplexSpiral


