# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False

cimport cython
import numpy as np
from cython.parallel import prange


cdef double[:] l1_grad(double[:] current_ticker_units, double l1_reg):
    cdef:
        int n = current_ticker_units.shape[0]
        int i
        double[:] out = np.zeros(n, dtype=np.float64)

    if l1_reg < 1e-7:
        return out

    for i in prange(n, nogil=True):
        out[i] = l1_reg if current_ticker_units[i] > 0 else -l1_reg
    return out


cdef double[:] l2_grad(double[:] current_ticker_units, double l2_reg):
    cdef:
        int n = current_ticker_units.shape[0]
        int i
        double[:] out = np.zeros(n, dtype=np.float64)
        double val

    if l2_reg < 1e-7:
        return out

    for i in prange(n, nogil=True):
        val = current_ticker_units[i]
        out[i] = l2_reg * val if val > 0 else -l2_reg * val
    return out


cdef double[:] ticker_unit_gradients(double[:, :] closing_values, double[:] current_portfolio_close,
                                     double[:] current_ticker_units, double alpha):
    cdef int n = closing_values.shape[0]
    cdef int num_tickers = closing_values.shape[1]

    cdef double[:] combined_close = get_combined_close(closing_values, current_ticker_units,
                                                       current_portfolio_close)

    cdef double[:, :] dr_db = get_dr_db(closing_values, combined_close, current_ticker_units)
    cdef double[:] dmu_db = np.mean(dr_db, axis=0)
    cdef double[:] dsigma_db = get_dsigma_db(combined_close, dr_db, dmu_db)

    cdef double[:] gradient = np.zeros(num_tickers, dtype=np.float64)
    cdef int i
    for i in prange(num_tickers, nogil=True):
        gradient[i] = dsigma_db[i] - alpha * dmu_db[i]
    return gradient


cdef double[:] get_dsigma_db(double[:] combined_close, double[:, :] dr_db, double[:] dmu_db):
    cdef int n = combined_close.shape[0]
    cdef int num_tickers = dr_db.shape[1]

    cdef double[:] r_t = np.log(combined_close[1:]) - np.log(combined_close[:(n - 1)])
    cdef double mean_r_t = (np.log(combined_close[n - 1]) - np.log(combined_close[0])) / n

    cdef double[:, :] out = np.zeros((n - 1, num_tickers), dtype=np.float64)

    cdef int i, j
    for i in prange(n - 1, nogil=True):
        for j in range(num_tickers):
            out[i, j] = (r_t[i] - mean_r_t) * (dr_db[i, j] - dmu_db[j])

    return np.mean(out, axis=0)


cdef double[:, :] get_dr_db(double[:, :] closing_values, double[:] combined_close, double[:] current_ticker_units):
    cdef int n = closing_values.shape[0]
    cdef int num_tickers = closing_values.shape[1]

    cdef double[:, :] close_ratio = get_close_ratio(closing_values, combined_close)

    cdef double[:, :] dr_db = np.zeros((n - 1, num_tickers), dtype=np.float64)
    cdef int i, j
    for i in prange(n - 1, nogil=True):
        for j in range(num_tickers):
            if current_ticker_units[j] > 0:
                dr_db[i, j] = close_ratio[i + 1, j] - close_ratio[i, j]
            else:
                dr_db[i, j] = 0.0
    return dr_db


cdef double[:] get_combined_close(double[:, :] closing_values, double[:] current_ticker_units,
                       double[:] current_portfolio_close):
    cdef int n = closing_values.shape[0]
    cdef int num_tickers = closing_values.shape[1]

    cdef double[:] out = np.zeros(n, dtype=np.float64)
    cdef double total
    cdef int i, k

    for i in prange(n, nogil=True):
        total = 0
        for k in range(num_tickers):
            total = total + closing_values[i, k] * current_ticker_units[k]
        out[i] = total + current_portfolio_close[i]
    return out


cdef double[:, :] get_close_ratio(double[:, :] closing_values, double[:] combined_close):
    cdef int n = closing_values.shape[0]
    cdef int num_tickers = closing_values.shape[1]

    cdef double[:, :] out = np.zeros((n, num_tickers), dtype=np.float64)

    cdef int i, j
    for i in prange(n, nogil=True):
        for j in range(num_tickers):
            out[i, j] = closing_values[i, j] / combined_close[i]
    return out
