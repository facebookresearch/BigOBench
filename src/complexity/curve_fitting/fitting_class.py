# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Code adapted from https://github.com/pberkes/big_O/blob/master/big_o/complexities.py
# Refer to original file for Licensing and copyright information

import numpy as np
import scipy
from scipy.optimize import curve_fit

class ComplexityClass(object):
    """
    Fits a complexity to a runtime curve or memory curve
    """
    def __init__(self):
        self.coeff = None

    def fit(
        self, 
        n, 
        t, 
        apply_constraints = True, 
        zero_out_first_value = True,
        piecewise_fit = True,
        max_time_x_value=3000,
    ):  
        
        n = np.asanyarray(n)
        t = np.asanyarray(t)
        assert np.argmin(t) == 0

        if zero_out_first_value:
            t = t - min(t)

        x_linear = Linear()._transform_n(n)
        x = self._transform_n(n)
        y = self._transform_time(t)

        if zero_out_first_value:
            x_linear = x_linear - np.array([0, x_linear[0, 1]])

            if (not isinstance(self, Constant)) and (not isinstance(self, Constant_2)):
                x = x - np.array([0, x[0, 1]])

        piecewise_coeff_list = []
        piecewise_x_list = []

        if apply_constraints:
            def inference_function(u_values, v_values):
                coeff,_ = scipy.optimize.nnls(
                    u_values, 
                    v_values
                )
                return coeff

        else:
            def inference_function(u_values, v_values):
                coeff, _, rank, s = np.linalg.lstsq(
                    u_values, 
                    v_values, 
                    rcond=-1
                )
                return coeff
            
        def wrapped_inference_function(u_values, v_values, zero_out_first_value, intercept_value):
            try:
                if zero_out_first_value:
                    coeff = inference_function(
                        u_values[:, 1:], 
                        v_values - intercept_value,
                    )
                    coeff = [intercept_value, coeff[0]]
                else:
                    coeff = inference_function(
                        u_values,
                        v_values,
                    )
            except:
                try:
                    if zero_out_first_value:
                        coeff = inference_function(
                            u_values[:, 1:], 
                            v_values - intercept_value,
                        )
                        coeff = [intercept_value, coeff[0]]
                    else:
                        coeff = inference_function(
                            u_values,
                            v_values,
                        )
                except:
                    coeff = [0, 0] 

            return coeff
        
        max_n_value_divided = 480
        
        for i, n_value in enumerate(n):
            if n_value > max_n_value_divided:
                break

        i = max(i, 1)

        if isinstance(self, Constant):
            if piecewise_fit and len(n[i:]) >= 2 and len(n[:i-1]) >= 2:
                coeff_1 = wrapped_inference_function(
                    x_linear[:i, :], 
                    y[:i], 
                    True, 
                    0
                )

                i -= 1
                
                coeff_2 = [y[i]]

                piecewise_coeff_list = [coeff_1, coeff_2]
                piecewise_x_list = [x_linear[:i, :], x[i:, :]]

            else:
                coeff = [y[0]]

                piecewise_coeff_list = [coeff]
                piecewise_x_list = [x[:, :]]

        elif isinstance(self, Constant_2):

            max_n_value_divided = 20
            
            for i, n_value in enumerate(n):
                if n_value > max_n_value_divided:
                    break

            i = max(i, 1)
                
            if piecewise_fit and len(n[i:]) >= 2 and len(n[:i-1]) >= 2:
                coeff_1 = wrapped_inference_function(
                    x_linear[:i, :], 
                    y[:i], 
                    True, 
                    0
                )

                i -= 1
                
                coeff_2 = [y[i]]

                piecewise_coeff_list = [coeff_1, coeff_2]
                piecewise_x_list = [x_linear[:i, :], x[i:, :]]

            else:
                coeff = [y[0]]

                piecewise_coeff_list = [coeff]
                piecewise_x_list = [x[:, :]]

        elif isinstance(self, Linear_2) and piecewise_fit:
            max_n_value_divided = 20
            
            for i, n_value in enumerate(n):
                if n_value > max_n_value_divided:
                    break

            i -= 1
            i = max(i, 1)

            if piecewise_fit and len(n[i:]) >= 2 and len(n[:i-1]) >= 2:
                
                coeff_1 = [(y[i] - y[0])/(x[i, 1] - x[0, 1])]
                coeff_1 = [y[0], coeff_1[0]]

                coeff_2 = wrapped_inference_function(
                    x[i:, :] - np.array([0, x[i, 1]]), 
                    y[i:], 
                    True, 
                    y[i], #coeff_1[1] * (x[i, 1] - x[0, 1])
                )

                # coeff_1 = [(coeff_2[1] * x[i, 1] + coeff_2[0] - y[0])/(x[i, 1] - x[0, 1])]
                # coeff_1 = [y[0], coeff_1[0]]
                
                piecewise_coeff_list = [coeff_1, coeff_2]
                piecewise_x_list = [x[:i, :], x[i:, :] - np.array([0, x[i, 1]])]

            else:
                coeff = wrapped_inference_function(
                    x[:, :], 
                    y[:], 
                    True, 
                    0, #coeff_1[1] * (x[i, 1] - x[0, 1])
                )
                        
                piecewise_coeff_list = [coeff]
                piecewise_x_list = [x]

        elif isinstance(self, Linear) and piecewise_fit and len(n[i:]) >= 2 and len(n[:i-1]) >= 2:
            i -= 1

            coeff_1 = [(y[i] - y[0])/(x[i, 1] - x[0, 1])]
            coeff_1 = [y[0], coeff_1[0]]

            coeff_2 = wrapped_inference_function(
                x[i:, :] - np.array([0, x[i, 1]]), 
                y[i:], 
                True, 
                y[i], #coeff_1[1] * (x[i, 1] - x[0, 1])
            )

            # coeff_1 = [(coeff_2[1] * x[i, 1] + coeff_2[0] - y[0])/(x[i, 1] - x[0, 1])]
            # coeff_1 = [y[0], coeff_1[0]]
            
            piecewise_coeff_list = [coeff_1, coeff_2]
            piecewise_x_list = [x[:i, :], x[i:, :] - np.array([0, x[i, 1]])]
                
        else:
            coeff = wrapped_inference_function(
                x[:, :], 
                y[:], 
                True, 
                0, #coeff_1[1] * (x[i, 1] - x[0, 1])
            )
                    
            piecewise_coeff_list = [coeff]
            piecewise_x_list = [x]

        assert sum(map(len, piecewise_x_list)) == len(n)

        self.piecewise_coeff_list = piecewise_coeff_list

        ref_t = self.compute(piecewise_x_list, n)
        residuals = np.sum((ref_t - t) ** 2)

        self.piecewise_coeff_list = [self.piecewise_coeff_list[-1]]
        max_time = self.compute([self._transform_n(np.array([max_time_x_value]))], np.array([max_time_x_value]))

        return residuals, ref_t, t, max_time[0], self.piecewise_coeff_list[-1][-1]

    def compute(self, piecewise_x_list, n):
        """ Compute the value of the fitted function at `n`. """        
        if len(self.piecewise_coeff_list) == 0:
            raise Exception()
        
        tot = np.zeros(len(n))

        assert sum(map(len, piecewise_x_list)) == len(n)
        assert len(piecewise_x_list) == len(self.piecewise_coeff_list)

        start_index = 0

        for coeff, x_list in zip(self.piecewise_coeff_list, piecewise_x_list):
            assert len(x_list[0]) == len(coeff)
            
            for i in range(len(coeff)):
                tot[start_index:start_index+len(x_list)] += coeff[i] * x_list[:, i]

            start_index += len(x_list)

        start_index == len(n)

        return self._inverse_transform_time(tot)

    def coefficients(self):
        """ Return coefficients in standard form. """
        if self.coeff is None:
            raise Exception()
        return self.coeff

    def __str__(self):
        prefix = '{}: '.format(self.__class__.__name__)

        if self.coeff is None:
            return prefix + 'not yet fitted'
        return prefix + self.format_str().format(
            *self.coefficients()) + ' (sec)'

    @classmethod
    def format_str(cls):
        """ Return a string describing the fitted function.

        The string must contain one formatting argument for each coefficient.
        """
        return 'FORMAT STRING NOT DEFINED'

    def _transform_n(self, n):
        """ Terms of the linear combination defining the complexity class.

        Output format: number of Ns x number of coefficients .
        """
        raise NotImplementedError()

    def _transform_time(self, t):
        """ Transform time as needed for fitting.
        (e.g., t->log(t)) for exponential class.
        """
        return t

    def _inverse_transform_time(self, t):
        """ Inverse transform time as needed for compute.
        (e.g., t->exp(t)) for exponential class.
        """
        return t

    def __hash__(self):
        return id(self)

class Constant(ComplexityClass):
    def _transform_n(self, n):
        return np.ones((len(n), 1))

    @classmethod
    def format_str(cls):
        return 'time = {:.2G}'
    
class Constant_2(ComplexityClass):
    def _transform_n(self, n):
        return np.ones((len(n), 1))

    @classmethod
    def format_str(cls):
        return 'time = {:.2G}'


class Linear(ComplexityClass):
    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n'
    

class Linear_2(ComplexityClass):
    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n'


class Quadratic(ComplexityClass):
    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n * n]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n^2'


class Cubic(ComplexityClass):
    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n ** 3]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n^3'


class Logarithmic(ComplexityClass):
    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), np.log(n)]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*log(n)'


class Polynomial(ComplexityClass):
    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), np.log(n)]).T

    def _transform_time(self, t):
        raise Exception('not implemented ?')
        return np.log(t)

    def _inverse_transform_time(self, t):
        raise Exception('not implemented ?')
        return np.exp(t)

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} * x^{:.2G}'

    def coefficients(self):
        """ Return coefficients in standard form. """
        # The polynomial is stored in the format
        # exp(a)*n^b where [a, b] are the coefficients
        # Technical full format is exp(a+b*ln(n))
        #
        # Standard form is a*n^b
        if self.coeff is None:
            raise Exception()

        a, b = self.coeff
        return np.exp(a), b


class Exponential(ComplexityClass):
    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n]).T

    def _transform_time(self, t):
        raise Exception('not implemented ?')
        return np.log(t)

    def _inverse_transform_time(self, t):
        raise Exception('not implemented ?')
        return np.exp(t)

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} * {:.2G}^n'

    def coefficients(self):
        """ Return coefficients in standard form. """
        # The polynomial is stored in the format
        # exp(a)*exp(b)^n where [a, b] are the coefficients
        # Technical full format is exp(a+b*n)
        #
        # Standard form is a*b^n
        if self.coeff is None:
            raise Exception()

        a, b = self.coeff
        return np.exp(a), np.exp(b)
    
class Linearithmic(ComplexityClass):
    inside_coef = 0.05
    inside_coef = 1
    # inside_coef = 0.0005

    # inside_coef = 0.0004

    # inside_coef = 0.0003

    def _transform_n(self, n):
        return np.vstack([np.ones(len(n)), n * np.log(self.inside_coef*n)]).T

    @classmethod
    def format_str(cls):
        return 'time = {:.2G} + {:.2G}*n*log(n)'