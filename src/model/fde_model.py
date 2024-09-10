import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve

class FDEModel:
    """
    Finite Difference Equation model for pricing derivatives using a stock process.
    """

    def __init__(self, num_time_steps, payoff_function):
        self.num_time_steps = int(num_time_steps)
        self.time_step = np.double(payoff_function.T) / num_time_steps
        self.payoff_function = payoff_function

    def price(self, lower_stock_price, upper_stock_price, num_stock_steps, scheme, stochastic_process, **kwargs):
        """
        Price the derivative for stock prices in the range [lower_stock_price, upper_stock_price]
        using the given finite difference scheme.
        """
        lower_stock_price = np.double(lower_stock_price)
        upper_stock_price = np.double(upper_stock_price)
        num_stock_steps = int(num_stock_steps)

        stock_prices = np.linspace(lower_stock_price, upper_stock_price, num_stock_steps + 1)
        stock_step = stock_prices[1] - stock_prices[0]
        terminal_values = self.payoff_function.terminal(stock_prices)

        scheme = scheme(stochastic_process, self.time_step, stock_step, stock_prices, **kwargs)

        for i in range(self.num_time_steps - 1, -1, -1):
            coupon_payment = self.payoff_function.coupon(self.payoff_function.T * i / self.num_time_steps)
            default_value = self.payoff_function.default(stock_prices)
            terminal_values, _ = scheme(i, terminal_values, default_value, coupon_payment, self.payoff_function.transient)

        return terminal_values
