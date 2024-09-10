import numpy as np

class BinomialTreeModel:
    """
    Implements a binomial tree model for pricing derivatives using a stock process.
    """

    def __init__(self, num_periods, time_increment, payoff_function):
        self.num_periods = int(num_periods)
        self.time_increment = np.double(time_increment)
        self.payoff_function = payoff_function

    def price(self, initial_stock_price, stochastic_process):
        """Calculates the price of the derivative for the initial stock price."""
        up_factor, down_factor, loss_factor, up_prob, down_prob, jump_prob = stochastic_process.get_binomial_params(self.time_increment)

        discount_factor = np.exp(-stochastic_process.risk_free_rate * self.time_increment)
        stock_prices = np.array([initial_stock_price * up_factor ** (self.num_periods - i) * down_factor ** i for i in range(self.num_periods + 1)])
        payoff_values = self.payoff_function.terminal(stock_prices)

        for i in range(self.num_periods - 1, -1, -1):
            stock_prices = stock_prices[:-1] / up_factor
            default_values = self.payoff_function.default(stock_prices * loss_factor)
            payoff_values = discount_factor * (payoff_values[:-1] * up_prob + payoff_values[1:] * down_prob + default_values * jump_prob)
            payoff_values = self.payoff_function.transient(i, stock_prices, payoff_values)  # Pass the time index `i`


        return payoff_values[0]
