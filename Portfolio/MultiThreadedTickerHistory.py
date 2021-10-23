import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import yfinance as yf

LOCK = threading.Lock()


def get_buffered_closing_for_tickers(tickers, start, end, max_workers=100):
    buffered_ticker_history = get_buffered_history_for_tickers(tickers, start, end, max_workers=max_workers)

    closing_prices = pd.concat([history.loc[:, 'Close'] for history in buffered_ticker_history.values()], axis=1)
    closing_prices.columns = list(buffered_ticker_history.keys())

    return closing_prices


def get_buffered_history_for_tickers(tickers, start, end, max_workers=100):
    """ Tickers only with values from the start to end dates will be kept.

        A warning will be raised if a ticker does not have values from the start
    """
    ticker_history = get_ticker_history(tickers, start, end, max_workers=max_workers)
    
    days, columns = get_asx_default_days_and_columns(start, end)
    buffered_ticker_history = {}
    invalid_tickers = []
    
    for ticker, history in ticker_history.items():
        if len(history) == 0:
            invalid_tickers.append(ticker)
        else:
            buffered_ticker_history[ticker] = buffer_history(history, days, columns)
            
    return buffered_ticker_history


def get_ticker_history(tickers, start_date, end_date, max_workers=100):
    ticker_history = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for ticker in tickers:
            future = executor.submit(update_ticker_history,
                                     ticker_history, ticker, start_date, end_date)
            futures.append(future)
            
        for future in futures:
            future.result()
            
    return ticker_history


def update_ticker_history(ticker_history, ticker_sym, start_date, end_date):
    ticker = yf.Ticker(ticker_sym)
    history = ticker.history(start=start_date, end=end_date)
    
    with LOCK:
        ticker_history[ticker_sym] = history
        

def get_asx_default_days_and_columns(start, end):
    asx_ticker_symbol = '^AXJO'

    asx_ticker = yf.Ticker(asx_ticker_symbol)
    asx_history = asx_ticker.history(start=start, end=end)

    days = asx_history.index
    columns = asx_history.columns

    return days, columns


def buffer_history(history, days, columns):
    buffered_history = pd.DataFrame(index=days, columns=columns)
    
    shared_days = days.intersection(set(history.index))
    
    buffered_history.loc[shared_days] = history.loc[shared_days]
    
    buffered_history.fillna(method='ffill', inplace=True)
    buffered_history.fillna(0, inplace=True)

    return buffered_history