import numpy as np
import pandas as pd

from functools import reduce
import warnings

import yfinance as yf

from datetime import date, timedelta
START_DATE = date(2017, 1, 5)
END_DATE = date(2021, 6, 10)


class Portfolio:
    count = 0

    def __init__(self, portfolio, start=START_DATE, end=END_DATE, ticker_prices=None, name=None):
        """
            Parameters
            ----------
            portfolio : dict of str - int
                The portfolio with keys as the ticker symbols, and values the shares of owned units
            start : datetime.date
                The start date
            end : datetime.date
                The end date
            ticker_prices : dict of str - pd.DataFrame, optional
                This essentially should be the cached ticker_history as given from yf.Ticker.history.
                If it is given then ticker_prices will be used to get the historic prices of tickers.
                Otherwise the prices will require the internet to get the historic prices
            name : str, optional
                The name for the portfolio
        """

        self.portfolio = portfolio
        self.start = start
        self.end = end

        tickers = list(self.portfolio.keys())
        buffered_ticker_history = self._get_buffered_history_for_tickers(tickers, start, end,
                                                                         ticker_prices=ticker_prices)

        weighted_history = [history * portfolio[ticker] for ticker, history in buffered_ticker_history.items()]

        self.portfolio_history = reduce(lambda x, y: x + y, weighted_history)
        self.close = self.portfolio_history['Close']

        returns = np.log(self.close.values[1:]/self.close.values[:-1]) * 100
        self.returns = pd.DataFrame(returns, index=self.portfolio_history.index[1:], columns=['Portfolio'])

        Portfolio.count += 1
        self.name = f'Portfolio {Portfolio.count}' if name is None else name

    @staticmethod
    def _get_buffered_history_for_tickers(tickers, start, end, ticker_prices=None):
        """ Tickers only with values from the start to end dates will be kept.

            A warning will be raised if a ticker does not have values from the start
        """
        days, columns = Portfolio._get_asx_default_days_and_columns(start, end, ticker_prices=ticker_prices)
        buffered_ticker_history = {}
        invalid_tickers = []
        for ticker in tickers:
            buffered_history = Portfolio._get_buffered_history(ticker, days, columns, start, end,
                                                               ticker_prices=ticker_prices)
            if buffered_history is None:
                invalid_tickers.append(ticker)
            else:
                buffered_ticker_history[ticker] = buffered_history

        if len(invalid_tickers) > 0:
            warnings.warn('Portfolio only works if the tickers have values for the start to end dates. These'
                          + f' tickers do not: {invalid_tickers}')
        return buffered_ticker_history

    @staticmethod
    def _get_asx_default_days_and_columns(start, end, ticker_prices=None):
        asx_ticker_symbol = '^AXJO'
        if ticker_prices is not None and asx_ticker_symbol in ticker_prices:
            asx_history = ticker_prices[asx_ticker_symbol]
        else:
            asx_ticker = yf.Ticker(asx_ticker_symbol)
            asx_history = asx_ticker.history(start=start, end=end)

        days = asx_history.index
        columns = asx_history.columns

        return days, columns

    @staticmethod
    def _get_buffered_history(ticker, days, columns, start, end, ticker_prices=None):
        if ticker_prices is not None and ticker in ticker_prices:
            history = ticker_prices[ticker]
        else:
            history = yf.Ticker(ticker).history(start=start, end=end)

        buffered_history = pd.DataFrame(index=days, columns=columns)
        shared_days = days.intersection(set(history.index))
        buffered_history.loc[shared_days] = history.loc[shared_days]
        buffered_history.fillna(method='ffill', inplace=True)

        is_close_na = buffered_history['Close'].isna()
        if is_close_na.values.sum() > 0:
            return None
        return buffered_history

    @property
    def log_intra_day_range(self):
        return 100 * np.log(self.portfolio_history['High'] / self.portfolio_history['Low'])

    @property
    def volatility_proxy_1(self):
        return abs(self.returns)

    @property
    def volatility_proxy_2(self):
        """ The intra-day """
        return np.sqrt(0.3607) * self.log_intra_day_range

    @property
    def volatility_proxy_3(self):
        """ This is really proxy 4 in the lecture slides, but im not going to be implementing proxy 3 """
        return np.exp(np.log(self.log_intra_day_range) - 0.43 + 0.29**2)

    def volatility_proxy(self, i):
        proxies = {1: self.volatility_proxy_1,
                   2: self.volatility_proxy_2,
                   3: self.volatility_proxy_3}
        if 1 <= i <= 3:
            return proxies[i]
        else:
            raise ValueError('Only proxy 1, 2, 3 is implemented')

    def add_ticker_to_portfolio(self, ticker, position_value, ticker_prices=None, name=None):
        """ Returns the new portfolio when the ticker is added

            The number of units of the new ticker is computed based on the position_value that one
            wants to take

            Parameters
            ----------
            ticker: str
            position_value: float/int
                The total market value (in dollars) one wants to take into the ticker
            ticker_prices: dict of str - pd.DataFrame, optional
                The historic prices of all tickers, with keys assumed to be the tickers and values
                DataFrames give by yf.Ticker.history
            name : str, optional
                The name for the new portfolio
        """

        # If ticker is a list, send it to the other method
        if isinstance(ticker, list):
            return ValueError('ticker can no longer be a list of string')

        new_portfolio = self.portfolio.copy()
        if ticker in new_portfolio:
            new_portfolio[ticker] += self._get_position(ticker, position_value, ticker_prices=ticker_prices)
        else:
            new_portfolio[ticker] = self._get_position(ticker, position_value, ticker_prices=ticker_prices)

        return Portfolio(new_portfolio, ticker_prices, name=name)

    def add_tickers_to_portfolio(self, tickers, market_values, ticker_prices=None, name=None):
        """ Returns the new portfolio when the new tickers are added

            Parameters
            ----------
            tickers : list of str
            market_values : np.array (or array like)
                The market value for the individual tickers
            ticker_prices : dict of str - pd.DataFrame, optional
                If not given, then the historic prices will be retrived from the internet
            name : str, optional
                The name of the new portfolio
        """
        new_portfolio = self.portfolio.copy()

        for ticker, market_value in zip(tickers, market_values):
            if ticker in new_portfolio:
                new_portfolio[ticker] += self._get_position(ticker, market_value, ticker_prices=ticker_prices)
            else:
                new_portfolio[ticker] = self._get_position(ticker, market_value, ticker_prices=ticker_prices)

        return Portfolio(new_portfolio, ticker_prices, name=name)

    def _get_position(self, ticker, market_value, ticker_prices=None):
        """ For a given ticker, return the number of own units given the total position value """
        if ticker_prices is not None and ticker in ticker_prices:
            latest_price = ticker_prices[ticker]['Close'].iloc[-1]
        else:
            start = self.end - timedelta(days=1)
            ticker_history = yf.Ticker(ticker).history(start=start, end=self.end)
            latest_price = ticker_history['Close'].iloc[-1]

        return int(market_value / latest_price)


class SimplifiedPortfolio:
    """ Simplified Portfolio only looks at the close column

        Attributes
        ----------
        portfolio : dict of str - int
            Keys the tickers, and values the number of units
        close : np.array
            The combined close of the given portfolio.
            Note, close is not an attribute of Portfolio
        returns : np.array
            The log returns
            Note, returns is a pd.DataFrame in Portfolio
        name : str
            The name of the class
    """
    count = 0

    def __init__(self, portfolio, start=START_DATE, end=END_DATE, ticker_prices=None, name=None):
        """
            Parameters
            ----------
            portfolio : dict of str - int
                The portfolio with keys as the ticker symbols, and values the number of owned units
        """

        self.portfolio = portfolio
        self.start = start
        self.end = end

        tickers = list(self.portfolio.keys())
        buffered_ticker_history = Portfolio._get_buffered_history_for_tickers(tickers, start, end,
                                                                              ticker_prices=ticker_prices)

        weighted_close = [history.loc[:, 'Close'] * portfolio[ticker]
                          for ticker, history in buffered_ticker_history.items()]

        self.close = reduce(lambda x, y: x + y, weighted_close).values
        self.returns = np.log(self.close[1:] / self.close[:-1]) * 100

        SimplifiedPortfolio.count += 1
        self.name = f'SimplifiedPortfolio {SimplifiedPortfolio.count}' if name is None else name

    def add_ticker_to_portfolio(self, ticker, market_value, ticker_prices=None, name=None):
        """ Returns the new portfolio when the ticker is added

            The number of units of the new ticker is computed based on the position_value that one
            wants to take

            Parameters
            ----------
            ticker: str
            market_value: float/int
                The total position (in dollars) one wants to take into the ticker
            ticker_prices: dict of str - pd.DataFrame, optional
                The historic prices of all tickers, with keys assumed to be the tickers and values
                DataFrames give by yf.Ticker.history
            name : str, optional
                The name of the new portfolio
        """

        new_portfolio = self.portfolio.copy()
        if ticker in new_portfolio:
            new_portfolio[ticker] += self._get_position(ticker, market_value, ticker_prices=ticker_prices)
        else:
            new_portfolio[ticker] = self._get_position(ticker, market_value, ticker_prices=ticker_prices)

        return SimplifiedPortfolio(new_portfolio, ticker_prices, name=name)

    def add_tickers_to_portfolio(self, tickers, market_values, ticker_prices=None, name=None):
        """ Returns the new portfolio when the ticker is added

            The number of units of the new ticker is computed based on the position_value that one
            wants to take

            Parameters
            ----------
            tickers: list of str
            market_values: list (or array like) of float/int
                The total position (in dollars) one wants to take into the ticker
            ticker_prices: dict of str - pd.DataFrame, optional
                The historic prices of all tickers, with keys assumed to be the tickers and values
                DataFrames give by yf.Ticker.history
            name : str, optional
                The name of the new portfolio
        """
        new_portfolio = self.portfolio.copy()

        for ticker, market_value in zip(tickers, market_values):
            if ticker in new_portfolio:
                new_portfolio[ticker] += self._get_position(ticker, market_value, ticker_prices=ticker_prices)
            else:
                new_portfolio[ticker] = self._get_position(ticker, market_value, ticker_prices=ticker_prices)

        return SimplifiedPortfolio(new_portfolio, ticker_prices, name=name)

    def _get_position(self, ticker, market_value, ticker_prices=None):
        """ For a given ticker, return the number of own units given the total position value """
        if ticker_prices is not None and ticker in ticker_prices:
            latest_price = ticker_prices[ticker]['Close'].iloc[-1]
        else:
            start = self.end - timedelta(days=1)
            ticker_history = yf.Ticker(ticker).history(start=start, end=self.end)
            latest_price = ticker_history['Close'].iloc[-1]

        return int(market_value / latest_price)