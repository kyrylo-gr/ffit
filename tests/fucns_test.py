from typing import Callable, NamedTuple, Tuple
import unittest


import smart_import

from scifit import Sin, SinParams, Lorentzian, LorentzianParams, Polynomial, PolynomialParams
from scifit.models import Fitter

import numpy as np




class FuncsTests(unittest.TestCase):
    """Test all functions"""
    def setUp(self) -> None:
        self.to_test_funcs = [Sin, Lorentzian, Polynomial]
        # super().__init__()
        
    def test_sin(self):
        """test sinus function"""
        sin_params = SinParams(
            amplitude=1,
            frequency=3,
            phase=np.pi/3,
            offset=.5
        )
        x, y_data = self.create_dataset(Sin().func, params=sin_params)
        f = Fitter(Sin)
        fm = f.sfit(x, y_data)
        
        for true_param, guess_param in zip(sin_params, SinParams(*fm.params).right_form()):
            self.assertAlmostEqual(
                true_param, guess_param,
                delta=.1, msg=f"Failed for sin_params = {sin_params}")
        
    @staticmethod
    def create_dataset(func: Callable, params: NamedTuple, start: int=0, end: int=10, points: int=1000) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(start, end, points)
        return x, func(x, *params)
        
        
        
    
if __name__ == '__main__':
    unittest.main()
