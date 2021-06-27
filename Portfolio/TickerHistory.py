import pandas as pd
import warnings

import yfinance as yf


def get_buffered_closing_for_tickers(tickers, start, end, ticker_prices=None):
    buffered_ticker_history = get_buffered_history_for_tickers(tickers, start, end, ticker_prices=ticker_prices)

    closing_prices = pd.concat([history.loc[:, 'Close'] for history in buffered_ticker_history.values()], axis=1)
    closing_prices.columns = list(buffered_ticker_history.keys())

    return closing_prices


def get_buffered_history_for_tickers(tickers, start, end, ticker_prices=None):
    """ Tickers only with values from the start to end dates will be kept.

        A warning will be raised if a ticker does not have values from the start
    """
    days, columns = get_asx_default_days_and_columns(start, end, ticker_prices=ticker_prices)
    buffered_ticker_history = {}
    invalid_tickers = []
    for ticker in tickers:
        buffered_history = get_buffered_history(ticker, days, columns, start, end,
                                                ticker_prices=ticker_prices)
        if buffered_history is None:
            invalid_tickers.append(ticker)
        else:
            buffered_ticker_history[ticker] = buffered_history

    if len(invalid_tickers) > 0:
        warnings.warn('Portfolio only works if the tickers have values for the start to end dates. These'
                      + ' tickers do not: ' + str(invalid_tickers))
    return buffered_ticker_history


def get_asx_default_days_and_columns(start, end, ticker_prices=None):
    asx_ticker_symbol = '^AXJO'
    if ticker_prices is not None and asx_ticker_symbol in ticker_prices:
        asx_history = ticker_prices[asx_ticker_symbol]
    else:
        asx_ticker = yf.Ticker(asx_ticker_symbol)
        asx_history = asx_ticker.history(start=start, end=end)

    days = asx_history.index
    columns = asx_history.columns

    return days, columns


def get_buffered_history(ticker, days, columns, start, end, ticker_prices=None):
    if ticker_prices is not None and ticker in ticker_prices:
        history = ticker_prices[ticker]
    else:
        history = yf.Ticker(ticker).history(start=start, end=end)

    buffered_history = pd.DataFrame(index=days, columns=columns)
    shared_days = days.intersection(set(history.index))
    buffered_history.loc[shared_days] = history.loc[shared_days]
    buffered_history.fillna(method='ffill', inplace=True)

    buffered_history.fillna(0, inplace=True)

    return buffered_history
