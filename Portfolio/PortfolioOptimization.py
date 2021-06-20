import numpy as np
import pandas as pd


class PortfolioOptimization:
    """ Here we have the un-constrained portfolio optimization, where the only constraint
        is that the number of units one can have is positive.
    """

    def __init__(self, tickers, closing_values, current_portfolio_close=None):
        """ Set closing prices of tickers / current portfolio

            Parameters
            ----------
            tickers : list of str - length j
                List of the ticker symbols to be optimized over
            closing_values : (n, j) np.array
                Closing values of the tickers
                Note that
            current_portfolio_close : (n,) np.array, optional
                The closing prices of your current portfolio. If one doesn't exist, then set to 0
        """

        self.tickers = tickers
        self.closing_values = closing_values
        self.current_portfolio_close = current_portfolio_close

        self.cost_for_tickers = closing_values[-1]

    def get_best_ticker_combination(self, max_position=5000, epochs=1000, alpha=1, learning_rate=0.1,
                                    start_units=5, sample_period=1, l1_reg=0, save_training_history=False):
        """ Returns the number of units per ticker that will minimise the cost function
            std^2 + alpha * mu
            where std is the standard deviation of the log-returns, and mu is the mean of the log-returns

            Parameters
            ----------
            tickers : list of str
                List of the tickers
            closing_values : np.array
                np.array of the closing prices of the tickers
            max_position : float
                The max position for 1 ticker
            sample_period : float between 0 and 1, optional
                The period of time to optimize the portfolio on
        """

        N = int(len(self.closing_values) * sample_period)
        in_sample_closing_values = self.closing_values[:N]
        if self.current_portfolio_close is None:
            in_sample_portfolio_close = 0
        else:
            in_sample_portfolio_close = self.current_portfolio_close[:N]

        max_unit_for_tickers = max_position // self.closing_values[-1]

        results = self._gradient_descent(in_sample_portfolio_close, in_sample_closing_values,
                                         max_unit_for_tickers, epochs=epochs, alpha=alpha,
                                         learning_rate=learning_rate, start_units=start_units,
                                         l1_reg=l1_reg, save_training_history=save_training_history)

        return self._convert_results_to_df(results, save_training_history)

    def _convert_results_to_df(self, results, save_training_history):
        ticker_combination = results[0] if save_training_history else results
        ticker_combination_as_df = pd.DataFrame(ticker_combination, index=self.tickers, columns=['Units'])
        ticker_combination_as_df['Cost'] = ticker_combination_as_df * self.cost_for_tickers[:, None]

        if save_training_history:
            history = results[1]
            epochs = len(history)

            history_as_df = pd.DataFrame(history, index=np.arange(epochs), columns=self.tickers)
            return ticker_combination_as_df, history_as_df
        return ticker_combination_as_df

    @staticmethod
    def _gradient_descent(current_portfolio_close, closing_values, max_unit_for_tickers, epochs=1000, alpha=1,
                          learning_rate=0.1, start_units=5, l1_reg=0, save_training_history=False):
        """ Peforms gradient descent to find the best unit combinations """
        num_of_tickers = closing_values.shape[1]

        current_ticker_units = np.ones(num_of_tickers) * start_units

        if save_training_history:
            history = np.zeros((epochs, num_of_tickers))

        # Perform Gradient Descent
        grad_squared = 0
        for _ in range(epochs):
            gradient = PortfolioOptimization._get_ticker_unit_gradients(current_portfolio_close, closing_values,
                                                                        current_ticker_units, alpha)
            l1_gradient = l1_reg * (current_ticker_units > 0).astype(int)

            # RMSprop
            grad_squared = 0.9 * grad_squared + 0.1 * gradient * gradient
            current_ticker_units -= gradient * (learning_rate / np.sqrt(grad_squared) + l1_gradient)

            # Clip
            current_ticker_units = np.clip(current_ticker_units, 0, max_unit_for_tickers)

            if save_training_history:
                history[_, :] = current_ticker_units

        if save_training_history:
            return current_ticker_units, history
        return current_ticker_units

    @staticmethod
    def _get_ticker_unit_gradients(current_portfolio_close, closing_values, current_ticker_units, alpha):
        combined_close = np.sum(closing_values * current_ticker_units, axis=1) \
                         + current_portfolio_close
        close_ratio = (closing_values / combined_close[:, None])

        dr_db = (close_ratio[1:] - close_ratio[:-1]) * (current_ticker_units > 0)
        dmu_db = np.mean(dr_db, axis=0)

        r_t = np.log(combined_close[1:]) - np.log(combined_close[:-1])
        dsigma_db = np.mean((r_t - np.mean(r_t))[:, None] * (dr_db - dmu_db[None, :]), axis=0)

        gradient = dsigma_db - alpha * dmu_db
        return gradient
