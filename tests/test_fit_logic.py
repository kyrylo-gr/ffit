import typing as _t
import unittest
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Tuple

from ffit.fit_logic import FitLogic
from ffit.fit_results import FitResult
from ffit.utils import FuncParamClass, convert_param_class

_T = _t.TypeVar("_T")


class LinearFitParam(_t.Generic[_T], FuncParamClass):
    """Line parameters.

    Attributes:
        offset (float)
        amplitude (float)
    """

    keys = ("slope", "intercept")

    slope: _T
    intercept: _T


class LinearFitResult(LinearFitParam, FitResult[LinearFitParam]):
    param_class = convert_param_class(LinearFitParam)


class TestLinearFit(FitLogic[LinearFitResult]):
    """Test fit class implementing linear fit y = mx + b."""

    _result_class = LinearFitResult

    @staticmethod
    def func(x, m, b):
        return m * x + b

    @staticmethod
    def _guess(x, y, **kwargs):
        del kwargs
        # Simple linear regression for initial guess
        m = np.cov(x, y)[0, 1] / np.var(x)
        b = np.mean(y) - m * np.mean(x)
        return m, b


class TestFitLogic(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.fit = TestLinearFit()
        # Generate test data with known parameters
        self.true_slope = 2.5
        self.true_intercept = 1.0
        self.x = np.linspace(0, 10, 100)
        self.y = self.true_slope * self.x + self.true_intercept
        # Add some noise
        self.noisy_y = self.y + np.random.normal(0, 0.1, self.x.shape)

    def test_basic_fit(self):
        """Test basic fitting functionality."""
        result = self.fit.fit(self.x, self.noisy_y)

        # Check if parameters are close to true values
        self.assertAlmostEqual(result.slope, self.true_slope, places=1)
        self.assertAlmostEqual(result.intercept, self.true_intercept, places=1)

        # Check if fitted function works
        y_fit = result.res_func(self.x)
        self.assertEqual(y_fit.shape, self.y.shape)

    def test_guess(self):
        """Test parameter guessing functionality."""
        guess_result = self.fit.guess(self.x, self.noisy_y)

        # Check if guess is reasonable (within 50% of true values)
        self.assertLess(
            abs(guess_result.slope - self.true_slope) / self.true_slope, 0.5
        )
        self.assertLess(
            abs(guess_result.intercept - self.true_intercept)
            / (self.true_intercept + 1e-10),
            0.5,
        )

    def test_mask(self):
        """Test fitting with mask."""
        # Create outliers
        y_with_outliers = self.noisy_y.copy()
        y_with_outliers[::10] = 100  # Add outliers

        # Create mask to exclude outliers
        mask = y_with_outliers < 50

        result = self.fit.fit(self.x, y_with_outliers, mask=mask)

        # Check if parameters are still close to true values despite outliers
        self.assertAlmostEqual(result.slope, self.true_slope, places=1)
        self.assertAlmostEqual(result.intercept, self.true_intercept, places=1)

    def test_array_fit(self):
        """Test array fitting functionality."""
        # Create multiple datasets
        n_datasets = 5
        y_array = np.array(
            [
                self.noisy_y + np.random.normal(0, 0.1, self.x.shape)
                for _ in range(n_datasets)
            ]
        )

        result = self.fit.array_fit(self.x, y_array)

        # Check shape of results
        self.assertEqual(result.res_array.shape, (n_datasets, 2))  # 2 parameters

        # Check if all fits are reasonable
        for params in result.res_array:
            slope, intercept = params
            self.assertLess(abs(slope - self.true_slope) / self.true_slope, 0.5)
            self.assertLess(
                abs(intercept - self.true_intercept) / (self.true_intercept + 1e-10),
                0.5,
            )

    def test_multi_dimensional_x(self):
        """Test fitting with multi-dimensional x values."""
        # Create 2D x and y data
        x_2d = np.stack([self.x for _ in range(3)])  # 3 x arrays
        y_2d = np.stack([self.noisy_y for _ in range(3)])  # 3 y arrays

        result = self.fit.array_fit(x_2d, y_2d)

        # Check if parameters are close to true values for all fits
        for params in result.res_array:
            self.assertAlmostEqual(params[0], self.true_slope, places=1)
            self.assertAlmostEqual(params[1], self.true_intercept, places=1)

    def test_axis_parameter(self):
        """Test fitting along different axes."""
        # Create 2D data with fit axis in middle
        y_array = np.stack([self.noisy_y for _ in range(3)])  # Shape: (3, 100)
        y_array = np.stack([y_array for _ in range(2)])  # Shape: (2, 3, 100)

        # Fit along different axes
        result_default = self.fit.array_fit(self.x, y_array)  # Default axis=-1
        result_axis1 = self.fit.array_fit(self.x, y_array, axis=2)

        # Check shapes
        self.assertEqual(result_default.res_array.shape, result_axis1.res_array.shape)

        # Check if parameters are reasonable for both cases
        for params in result_default.res_array.reshape(-1, 2):
            self.assertLess(abs(params[0] - self.true_slope) / self.true_slope, 0.5)
            self.assertLess(
                abs(params[1] - self.true_intercept) / (self.true_intercept + 1e-10),
                0.5,
            )

    def test_mp_array_fit(self):
        """Test multiprocessing array fit."""
        # Create multiple datasets
        n_datasets = 5
        y_array = np.array(
            [
                self.noisy_y + np.random.normal(0, 0.1, self.x.shape)
                for _ in range(n_datasets)
            ]
        )

        result = self.fit.mp_array_fit(self.x, y_array, n_jobs=2)

        # Check shape of results
        self.assertEqual(result.res_array.shape, (n_datasets, 2))

        # Check if all fits are reasonable
        for params in result.res_array:
            slope, intercept = params
            self.assertLess(abs(slope - self.true_slope) / self.true_slope, 0.5)
            self.assertLess(
                abs(intercept - self.true_intercept) / (self.true_intercept + 1e-10),
                0.5,
            )

    async def test_async_array_fit(self):
        """Test async array fit."""
        # Create multiple datasets
        n_datasets = 5
        y_array = np.array(
            [
                self.noisy_y + np.random.normal(0, 0.1, self.x.shape)
                for _ in range(n_datasets)
            ]
        )

        result = await self.fit.async_array_fit(self.x, y_array)

        # Check shape of results
        self.assertEqual(result.res_array.shape, (n_datasets, 2))

        # Check if all fits are reasonable
        for params in result.res_array:
            slope, intercept = params
            self.assertLess(abs(slope - self.true_slope) / self.true_slope, 0.5)
            self.assertLess(
                abs(intercept - self.true_intercept) / (self.true_intercept + 1e-10),
                0.5,
            )

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid x and y shapes
        with self.assertRaises(ValueError):
            self.fit.fit(self.x, self.noisy_y[:50])  # Mismatched lengths

        # Test invalid mask shape
        with self.assertRaises(ValueError):
            self.fit.fit(self.x, self.noisy_y, mask=np.ones(50))  # Wrong mask length

        # Test invalid axis
        y_array = np.stack([self.noisy_y for _ in range(3)])
        with self.assertRaises(ValueError):
            self.fit.array_fit(self.x, y_array, axis=3)  # Invalid axis

    def test_bootstrapping(self):
        """Test bootstrapping functionality."""
        result = self.fit.bootstrapping(self.x, self.noisy_y, num_of_permutations=100)

        # Check if parameters are close to true values
        self.assertAlmostEqual(result.slope, self.true_slope, places=1)
        self.assertAlmostEqual(result.intercept, self.true_intercept, places=1)

        # Check if standard errors are reasonable (non-zero but not too large)
        self.assertGreater(result.stderr[0], 0)  # slope error
        self.assertGreater(result.stderr[1], 0)  # intercept error
        self.assertLess(
            result.stderr[0] / abs(self.true_slope), 0.5
        )  # relative error < 50%
        self.assertLess(result.stderr[1] / abs(self.true_intercept + 1e-10), 0.5)


if __name__ == "__main__":
    unittest.main()
