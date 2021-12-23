# cython: language_level=3

cdef double[:] l1_grad(double[:] current_ticker_units, double l1_reg)

cdef double[:] l2_grad(double[:] current_ticker_units, double l2_reg)

cdef double[:] ticker_unit_gradients(double[:, :] closing_values, double[:] current_portfolio_close,
                                     double[:] current_ticker_units, double alpha)
