import numpy as np
import pandas as pd
import warnings

from .MultiThreadedTickerHistory import get_buffered_closing_for_tickers
from .ValueChecks import check_for_incomplete_ticker_closing

from datetime import date

START_DATE = date(2017, 1, 5)
END_DATE = date(2021, 6, 10)


class ConstrainedPortfolioOptimization:
    """ In the Constrained Portfolio Optimization the aim is now to minimise(variance - alpha * mean)
        under the condition that we only have a fixed amount we can invest in. So

        sum(b_j * C_j) = max_portfolio_value, and  b_j > 0

        where max_portfolio_value is the max amount to invest in, and C_j are the last closing prices
        of the j-th stock.
    """

    def __init__(self, tickers, closing_values=None, start=START_DATE, end=END_DATE,
                 current_portfolio_close=None):
        """ Set closing prices of tickers / current portfolio

            Parameters
            ----------
            tickers : list of str - length j
                List of the ticker symbols to be optimized over
            closing_values : (n, j) np.array, optional
                Closing values of the tickers
                If not given, then it will be retrieved from the internet
            start : datetime.date
            end : datetime.date
            current_portfolio_close : (n,) np.array, optional
                The closing prices of your current portfolio. If one doesn't exist, then set to 0
        """

        unique_tickers = list(set(tickers))
        if closing_values is not None:
            self.closing_values = closing_values
            self.tickers = tickers
        else:
            closing_values = get_buffered_closing_for_tickers(unique_tickers, start, end)

            self.closing_values = closing_values.values
            self.tickers = list(closing_values.columns)

        check_for_incomplete_ticker_closing(self.tickers, self.closing_values)

        self.current_portfolio_close = current_portfolio_close

        self.cost_for_tickers = self.closing_values[-1]

    def get_best_ticker_combination(self, max_portfolio_value, epochs=1000,
                                    alpha=1.0, learning_rate=0.1, start_units=5, sample_period=1.0,
                                    l1_reg=0.0, l2_reg=0.0, save_training_history=False):
        """ Returns the number of units per ticker that will minimise the cost function
            std^2 - alpha * mu
            where std is the standard deviation of the log-returns, and mu is the mean of the log-returns

            Parameters
            ----------
            max_portfolio_value : float
                The max amount to invest in
            epochs : int, optional
                Number of times to perform gradient descent
            alpha : float, optional
                Weight on the averaged returns
            learning_rate : float, optional
                The learning rate
            start_units : int, optional
                The number of units to start the gradient descent
            sample_period : float between 0 and 1, optional
                The period of time to optimize the portfolio on
            l1_reg : float, optional
                L1-Regularization strength on the number of units
            l2_reg : float, optional
                L2-Regularization strength on the number of units
            save_training_history : bool, optional
                If true then the weights on the tickers will be returned as well
        """

        N = int(len(self.closing_values) * sample_period)
        in_sample_closing_values = self.closing_values[:N]

        if self.current_portfolio_close is None:
            in_sample_portfolio_close = 0
        else:
            in_sample_portfolio_close = self.current_portfolio_close[:N]

        results = self._projected_gradient_descent(in_sample_portfolio_close, in_sample_closing_values,
                                                   max_portfolio_value, self.cost_for_tickers,
                                                   epochs=epochs, alpha=alpha, learning_rate=learning_rate,
                                                   start_units=start_units, l1_reg=l1_reg, l2_reg=l2_reg,
                                                   save_training_history=save_training_history)

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
    def _projected_gradient_descent(current_portfolio_close, closing_values, max_portfolio_value,
                                    cost_for_tickers, epochs=1000, alpha=1.0, learning_rate=0.1, start_units=1,
                                    l1_reg=0.0, l2_reg=0.0, save_training_history=False, e=1e-7):

        CPO = ConstrainedPortfolioOptimization

        num_of_tickers = closing_values.shape[1]
        current_ticker_units = np.ones(num_of_tickers) * start_units

        if save_training_history:
            history = np.zeros((epochs, num_of_tickers))

        # Perform Projected Gradient Descent
        grad_squared = 0
        for _ in range(epochs):
            gradient = CPO._get_ticker_unit_gradients(closing_values, current_portfolio_close,
                                                      current_ticker_units, alpha)
            l1_gradient = l1_reg * (current_ticker_units > 0).astype(int)
            l2_gradient = l2_reg * current_ticker_units

            # RMSprop - Normal gradient descent tends to take a very long time, can probably
            # use Adam for quicker convergence
            grad_squared = 0.9 * grad_squared + 0.1 * gradient * gradient
            current_ticker_units -= learning_rate * (gradient / (np.sqrt(grad_squared) + e) + l1_gradient + l2_gradient)

            # Project back into space
            current_ticker_units = CPO._project_weights_into_constraint(current_ticker_units,
                                                                        max_portfolio_value,
                                                                        cost_for_tickers)
            if save_training_history:
                history[_, :] = current_ticker_units

        if save_training_history:
            return current_ticker_units, history
        return current_ticker_units

    @staticmethod
    def _get_ticker_unit_gradients(closing_values, current_portfolio_close, current_ticker_units,
                                   alpha):

        combined_close = np.sum(closing_values * current_ticker_units, axis=1) \
                         + current_portfolio_close
        close_ratio = (closing_values / combined_close[:, None])

        dr_db = (close_ratio[1:] - close_ratio[:-1]) * (current_ticker_units > 0)
        dmu_db = np.mean(dr_db, axis=0)

        r_t = np.log(combined_close[1:]) - np.log(combined_close[:-1])
        dsigma_db = np.mean((r_t - np.mean(r_t))[:, None] * (dr_db - dmu_db[None, :]), axis=0)

        gradient = dsigma_db - alpha * dmu_db
        return gradient

    @staticmethod
    def _project_weights_into_constraint(current_weights, total_amount, closing_prices):
        CPO = ConstrainedPortfolioOptimization

        # These are the max units of a ticker one can have if all other units are set to 0
        plane_end_points = total_amount / closing_prices

        count = 0
        max_iterations = len(current_weights)

        projected_weights = current_weights

        # The count is here to prevent infinite while loops
        while ((not CPO._is_valid_weights(projected_weights, total_amount, closing_prices))
               and count < max_iterations + 1):
            mask = projected_weights > 0
            projected_weights = CPO._project_weights(projected_weights * mask, closing_prices * mask,
                                                     plane_end_points * mask)
            count += 1

        if count == max_iterations + 1:
            warnings.warn('Max iteration reached, possibly didnt converge')
        return projected_weights

    @staticmethod
    def _is_valid_weights(weights, total_amount, closing_prices, e=1e-6):
        """ Weights are only valid if they are all positive, and the weighted sum is = the total amount """
        return np.all(weights > -e) and abs(np.sum(weights * closing_prices) - total_amount) < e

    @staticmethod
    def _project_weights(current_weights, closing_prices, plane_end_points):
        if np.all(plane_end_points <= 0):
            return np.zeros(len(plane_end_points))

        non_zero_index = np.argwhere(plane_end_points > 0).flatten()[0]

        point_on_plane = np.zeros(plane_end_points.shape)
        point_on_plane[non_zero_index] = plane_end_points[non_zero_index]

        return ConstrainedPortfolioOptimization._closest_vector_to_hyperplane(current_weights,
                                                                              closing_prices, point_on_plane)

    @staticmethod
    def _closest_vector_to_hyperplane(y, a, p):
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
        return y - (np.dot(y - p, a) / np.dot(a, a)) * a
