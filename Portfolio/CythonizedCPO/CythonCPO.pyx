# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
import warnings

from libc.math cimport sqrt
from cython.parallel cimport prange

cimport utils
from CPOGradients cimport l1_grad, l2_grad, ticker_unit_gradients


cdef class RMSProp:
    cdef double[:] grad_squared
    cdef double learning_rate
    cdef int n

    def __init__(self, n, learning_rate):
        self.grad_squared = np.zeros(n, dtype=np.float64)

        self.learning_rate = learning_rate
        self.n = n

    cdef double[:] step(self, double[:] grad, double e=1e-7):
        cdef int i
        cdef double[:] out = np.zeros(self.n, dtype=np.float64)

        for i in prange(self.n, nogil=True):
            self.grad_squared[i] = 0.9 * self.grad_squared[i] + 0.1 * (grad[i] ** 2)

            out[i] = self.learning_rate * (grad[i] / (sqrt(self.grad_squared[i]) + e))

        return out


cdef class CythonCPO:
    cdef:
        double alpha, learning_rate, l1_reg, l2_reg
        bint early_stopping, save_training_history

    def __init__(self, double alpha=1.0, double learning_rate=0.1, double l1_reg=0.0, double l2_reg=0.0,
                 bint early_stopping=False, bint save_training_history=False):
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.early_stopping = early_stopping
        self.save_training_history = save_training_history


    def projected_gradient_descent(self, double[:] current_portfolio_close, double[:, :] closing_values,
                                   double max_portfolio_value, double[:] cost_for_tickers,
                                   int epochs=1000, int start_units=1):
        cdef:
            int num_of_tickers = closing_values.shape[1]
            int i, epoch
            double[:] current_ticker_units = np.ones(num_of_tickers, dtype=np.float64) * start_units
            RMSProp optimizer = RMSProp(num_of_tickers, self.learning_rate)
            double[:] gradient = np.zeros(num_of_tickers, dtype=np.float64)
            double[:] step = np.zeros(num_of_tickers, dtype=np.float64)

        for epoch in range(epochs):
            gradient = self.get_gradient(closing_values, current_portfolio_close, current_ticker_units)
            step = optimizer.step(gradient)

            current_ticker_units = utils.subtract(current_ticker_units, step)

            # Project back into space
            current_ticker_units = self.project_weights_into_constraint(current_ticker_units,
                                                                        max_portfolio_value,
                                                                        cost_for_tickers)
        return current_ticker_units


    cdef double[:] get_gradient(self, double[:, :] closing_values, double[:] current_portfolio_close,
                                double[:] current_ticker_units):
        cdef double[:] l1 = l1_grad(current_ticker_units, self.l1_reg)
        cdef double[:] l2 = l2_grad(current_ticker_units, self.l2_reg)
        cdef double[:] grad = ticker_unit_gradients(closing_values, current_portfolio_close,
                                                    current_ticker_units, self.alpha)

        return utils.add(utils.add(l1, l2), grad)


    def project_weights_into_constraint(self, double[:] current_weights, double total_amount, double[:] closing_prices):
        # These are the max units of a ticker one can have if all other units are set to 0
        cdef:
            double[:] plane_end_points = utils.scalar_divide(total_amount, closing_prices)
            int count = 0
            int max_iterations = current_weights.shape[0]
            double[:] projected_weights = np.zeros(current_weights.shape[0], dtype=np.float64)
            double[:] mask = np.zeros(current_weights.shape[0], dtype=np.float64)

        projected_weights[:] = current_weights

        # The count is here to prevent infinite while loops
        while ((not CythonCPO.is_valid_weights(projected_weights, total_amount, closing_prices))
               and count < max_iterations + 1):
            mask = CythonCPO.is_positive(projected_weights)
            projected_weights = CythonCPO.project_weights(utils.hamard_prod(projected_weights, mask),
                                                          utils.hamard_prod(closing_prices, mask),
                                                          utils.hamard_prod(plane_end_points, mask))
            count += 1

        if count == max_iterations + 1:
            warnings.warn('Max iteration reached, possibly didnt converge')
        return projected_weights

    @staticmethod
    cdef double[:] is_positive(double[:] projected_weights):
        cdef:
            int n = projected_weights.shape[0]
            int i
            double[:] out = np.zeros(n, dtype=np.float64)

        for i in prange(n, nogil=True):
            out[i] = 1.0 if projected_weights[i] > 0.0 else 0.0
        return out

    @staticmethod
    cdef bint is_valid_weights(double[:] weights, double total_amount, double[:] closing_prices, double e=1e-6):
        """ Weights are only valid if they are all positive, and the weighted sum is = the total amount """
        cdef int n = weights.shape[0]
        cdef int i

        # Check weights are all positive
        for i in range(n):
            if weights[i] < -e:
                return False

        # Check weighted sum
        return abs(utils.dot(weights, closing_prices) - total_amount) < e

    @staticmethod
    cdef double[:] project_weights(double[:] current_weights, double[:] closing_prices, double[:] plane_end_points):
        cdef:
            int n = current_weights.shape[0]
            int non_zero_index = -1
            int i
            double[:] point_on_plane = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if plane_end_points[i] > 0:
                non_zero_index = i
                break

        if non_zero_index < 0:
            return point_on_plane

        point_on_plane[non_zero_index] = plane_end_points[non_zero_index]
        return CythonCPO.closest_vector_to_hyperplane(current_weights, closing_prices, point_on_plane)

    @staticmethod
    cdef double[:] closest_vector_to_hyperplane(double[:] y, double[:] a, double[:] p):
        """ The hyper plane assumed to be of the form np.dot(a, x - p) = d

            Parameters
            ----------
            y : np.array
                Arbitrary point
            a : np.array
                The Normal vector
            p : np.array
                Point on the hyper plane
        """
        cdef int n = a.shape[0]
        cdef double c = utils.dot(utils.subtract(y, p), a) / utils.dot(a, a)

        return utils.subtract(y, utils.scalar_mul(c, a))

    def callbacks(self):
        pass
