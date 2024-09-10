import numpy as np

class Payoff:
    """
    Base class for a generic payoff.
    """

    def __init__(self, maturity):
        self.T = maturity  # Time to maturity

    def default(self, stock_prices):
        """Default payoff function."""
        return np.zeros_like(stock_prices)

    def terminal(self, stock_prices):
        """Payoff at maturity."""
        return np.zeros_like(stock_prices)

    def transient(self, stock_prices, values):
        """Payoff at any time before maturity."""
        return values


class EuropeanCallOption(Payoff):
    """
    European call option payoff.
    """

    def __init__(self, maturity, strike_price):
        super().__init__(maturity)
        self.strike_price = strike_price

    def terminal(self, stock_prices):
        """Payoff at maturity: max(S - K, 0)."""
        return np.maximum(stock_prices - self.strike_price, 0)


class AmericanCallOption(EuropeanCallOption):
    """
    American call option payoff.
    """

    def transient(self, stock_prices, values):
        """Payoff at any time before maturity: max(S - K, V)."""
        return np.maximum(stock_prices - self.strike_price, values)
