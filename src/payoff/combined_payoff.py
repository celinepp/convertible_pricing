from .standard_payoff import Payoff

class PayoffCombination(Payoff):
    """
    Combines multiple payoff structures into one.
    """

    def __init__(self, payoffs):
        self.payoffs = payoffs
        self.T = max(payoff.T for payoff in self.payoffs)

    def terminal(self, stock_prices):
        """Combines terminal payoffs from multiple structures."""
        return sum(payoff.terminal(stock_prices) for payoff in self.payoffs)

    def default(self, stock_prices):
        """Combines default payoffs from multiple structures."""
        return sum(payoff.default(stock_prices) for payoff in self.payoffs)

    def transient(self, time, stock_prices, values):
        """
        Payoff during transient (non-terminal) time.
        Applies the transient payoff for each combined payoff structure.
        """
        for payoff in self.payoffs:
            values = payoff.transient(time, stock_prices, values)
        return values

    def coupon(self, time):
        """Combines coupon payments from all payoffs in the combination."""
        return sum(payoff.coupon(time) for payoff in self.payoffs if hasattr(payoff, 'coupon'))
