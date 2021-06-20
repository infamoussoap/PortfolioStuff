import numpy as np
import pandas as pd
from functools import reduce

class Portfolio:
    count = 0

    def __init__(self, portfolio, ticker_prices, name=None):
        """
            Parameters
            ----------
            portfolio : dict of str - int
                The portfolio with keys as the ticker symbols, and values the number of owned units
        """

        self.portfolio = portfolio

        BANNED_TICKER_FROM_PORTFOLIO = ['CLNE']
        weighted_prices = [ticker_prices[ticker] * weight for ticker, weight in portfolio.items()
                           if ticker not in BANNED_TICKER_FROM_PORTFOLIO]

        self.portfolio_prices = reduce(lambda x, y: x + y, weighted_prices)

        self.close = self.portfolio_prices['Close']

        returns = np.log(self.close.values[1:]/self.close.values[:-1]) * 100
        self.returns = pd.DataFrame(returns,
                                    index=self.portfolio_prices.index[1:],
                                    columns=['Portfolio'])

        Portfolio.count += 1
        self.name = f'Portfolio {Portfolio.count}' if name is None else name

    @property
    def log_intra_day_range(self):
        return 100 * np.log(self.portfolio_prices['High'] / self.portfolio_prices['Low'])

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

    def add_ticker_to_portfolio(self, ticker, position_value, ticker_prices, name=None):
        """ Returns the new portfolio when the ticker is added

            The number of units of the new ticker is computed based on the position_value that one
            wants to take

            Parameters
            ----------
            ticker: str or list
            position_value: float/int or list
                The total position (in dollars) one wants to take into the ticker
            ticker_prices: dict of str - pd.DataFrame
                The historic prices of all tickers, with keys assumed to be the tickers and values
                DataFrames give by yf.Ticker.history
        """

        # If ticker is a list, send it to the other method
        if isinstance(ticker, list):
            if not isinstance(position_value, list):
                position_value = [position_value for _ in range(len(ticker))]
            return self._add_tickers_to_portfolio(ticker, position_value, ticker_prices, name=name)


        new_portfolio = self.portfolio.copy()
        if ticker in new_portfolio:
            new_portfolio[ticker] += self._get_position(ticker, position_value, ticker_prices)
        else:
            new_portfolio[ticker] = self._get_position(ticker, position_value, ticker_prices)

        return Portfolio(new_portfolio, ticker_prices, name=name)

    def _add_tickers_to_portfolio(self, tickers, position_values, ticker_prices, name=None):
        """ tickers now assumed to be a list, and position_values will be a list as well """
        new_portfolio = self.portfolio.copy()

        for ticker, position_value in zip(tickers, position_values):
            if ticker in new_portfolio:
                new_portfolio[ticker] += self._get_position(ticker, position_value, ticker_prices)
            else:
                new_portfolio[ticker] = self._get_position(ticker, position_value, ticker_prices)

        return Portfolio(new_portfolio, ticker_prices, name=name)

    @staticmethod
    def _get_position(ticker, position_value, ticker_prices):
        """ For a given ticker, return the number of own units given the total position value """

        latest_price = ticker_prices[ticker]['Close'].iloc[-1]
        return int(position_value/latest_price)


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

    def __init__(self, portfolio, ticker_prices, name=None):
        """
            Parameters
            ----------
            portfolio : dict of str - int
                The portfolio with keys as the ticker symbols, and values the number of owned units
        """

        self.portfolio = portfolio

        BANNED_TICKER_FROM_PORTFOLIO = ['CLNE']
        weighted_close = [ticker_prices[ticker].loc[:, 'Close'] * weight for ticker, weight in portfolio.items()
                          if ticker not in BANNED_TICKER_FROM_PORTFOLIO]

        self.close = reduce(lambda x, y: x + y, weighted_close).values
        self.returns = np.log(self.close[1:]/self.close[:-1]) * 100

        SimplifiedPortfolio.count += 1
        self.name = f'SimplifiedPortfolio {SimplifiedPortfolio.count}' if name is None else name

    def add_ticker_to_portfolio(self, ticker, position_value, ticker_prices, name=None):
        """ Returns the new portfolio when the ticker is added

            The number of units of the new ticker is computed based on the position_value that one
            wants to take

            Parameters
            ----------
            ticker: str or list
            position_value: float/int or list
                The total position (in dollars) one wants to take into the ticker
            ticker_prices: dict of str - pd.DataFrame
                The historic prices of all tickers, with keys assumed to be the tickers and values
                DataFrames give by yf.Ticker.history
        """

        # If ticker is a list, send it to the other method
        if isinstance(ticker, list):
            if not isinstance(position_value, list):
                position_value = [position_value for _ in range(len(ticker))]
            return self._add_tickers_to_portfolio(ticker, position_value, ticker_prices, name=name)


        new_portfolio = self.portfolio.copy()
        if ticker in new_portfolio:
            new_portfolio[ticker] += self._get_position(ticker, position_value, ticker_prices)
        else:
            new_portfolio[ticker] = self._get_position(ticker, position_value, ticker_prices)

        return SimplifiedPortfolio(new_portfolio, ticker_prices, name=name)

    def _add_tickers_to_portfolio(self, tickers, position_values, ticker_prices, name=None):
        """ tickers now assumed to be a list, and position_values will be a list as well """
        new_portfolio = self.portfolio.copy()

        for ticker, position_value in zip(tickers, position_values):
            if ticker in new_portfolio:
                new_portfolio[ticker] += self._get_position(ticker, position_value, ticker_prices)
            else:
                new_portfolio[ticker] = self._get_position(ticker, position_value, ticker_prices)

        return SimplifiedPortfolio(new_portfolio, ticker_prices, name=name)

    @staticmethod
    def _get_position(ticker, position_value, ticker_prices):
        """ For a given ticker, return the number of own units given the total position value """

        latest_price = ticker_prices[ticker]['Close'].iloc[-1]
        return int(position_value/latest_price)
