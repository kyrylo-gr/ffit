---
title: Release Notes
---

## 1.1.0 (2024-12-25) - Major Refactoring

Key Changes to `FitResult`:

- `FitResult` is no longer a subclass of `tuple`. To achieve similar behavior and unpack `res` and `res_func`, use the new method `fit_result.res_and_func()`.
- All parameters are now directly accessible as attributes. Accessing parameters via `fit.res` is still possible but can be deprecated in the future.
- The `res` as np.ndarray is now accessible via `fit.res_array`.
- The `FitArrayResult` structure has been refactored and is now unified with `FitResult`.
- Results in `FitResult` are guaranteed to never be `None`. If the fit fails, each parameter will return `np.nan`, and the `success` attribute will be set to `False`.

Other Changes:

- `array_fit` can now be applied to any axes, offering greater flexibility.
- Each implemented fit function now has a dedicated result class with predefined slots and custom methods.
- The `leastsq` method has been removed. Without specifying an exact function, the `FitResult` for classical `leastsq` becomes useless. Use `curve_fit` instead, specifying `method='leastsq'`.

Normally these changes only affect you if you unpack the `FitResult` before. We apologize for any inconvenience, but we believe these changes will improve the package's usability in the long run and should be made as soon as possible.

## 1.0.0 (2024-11-16)

- Automatic documentation generation for each function.
- Automated testing for all functions.
- New guide available for creating custom functions.

## 0.2.0 (2024-08-28)

- Beta version of the package.
