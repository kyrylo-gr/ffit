---
title: ExpDecayingCos
---

### Simples example

```
>>> from ffit.funcs.exp_decaying_cos import ExpDecayingCos

# Call the fit method with x and y data.
>>> fit_result = ExpDecayingCos().fit(x, y)

# The result is a FitResult object that can be unpacked.
>>> res, res_func = fit_result

# One can combine multiple calls in one line.
>>> res = ExpDecayingCos().fit(x, y, guess=[1, 2, 3, 4]).plot(ax).res
```

<!-- prettier-ignore -->
::: ffit.funcs.exp_decaying_cos.ExpDecayingCos
