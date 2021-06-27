import numpy as np
import warnings


def check_for_incomplete_ticker_closing(tickers, ticker_closing_values):
    """
        Parameters
        ----------
        tickers : list[str]
            Length j
        ticker_closing_values : (n, j) np.array
    """
    if np.any(ticker_closing_values < 1e-7):
        incomplete_ticker_histories = np.any(ticker_closing_values < 1e-7, axis=0)
        incomplete_tickers = [ticker
                              for (is_incomplete, ticker) in zip(incomplete_ticker_histories, tickers)
                              if is_incomplete]

        warnings.warn('Tickers with incomplete historic price may overestimate the calculated volatility.'
                      f' The following tickers are incomplete: {incomplete_tickers}.')
