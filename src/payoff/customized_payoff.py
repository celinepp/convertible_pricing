import numpy as np
from .standard_payoff import Payoff

class ConvertibleBondAnnuity(Payoff):
    """Custom annuity where the coupon is paid as part of the nominal value."""

    def __init__(self, maturity, times, coupon_amount, nominal_value):
        super().__init__(maturity)
        self.T = maturity
        self.times = times
        self.coupon_amount = np.double(coupon_amount)
        self.nominal_value = np.double(nominal_value)

    def coupon(self, time):
        """Put options usually don't have coupon payments, return 0."""
        return 0.0

    def terminal(self, stock_prices):
        """Nominal value of bond with final coupon payment if applicable."""
        terminal_value = np.ones_like(stock_prices) * self.nominal_value
        if self.T in self.times:
            terminal_value += self.coupon_amount
        return terminal_value
    
    def transient(self, time, stock_prices, values):
        """Returns the values without changing, as the annuity doesn't change during non-terminal times."""
        return values


class ConvertibleBondPutOption(Payoff):
    """Customized put option where the strike price is adjusted for outstanding coupon."""

    def __init__(self, maturity, base_strike_price, annuity):
        super().__init__(maturity)
        self.T = maturity
        self.base_strike_price = base_strike_price
        self.annuity = annuity
    def coupon(self, time):
        """Put options usually don't have coupon payments, return 0."""
        return 0.0
    def strike(self, t):
        """Adjusts the strike price based on accrued coupons."""
        ti = 0
        acc_coupon = 0
        for payment_time in self.annuity.times:
            if payment_time > t:
                acc_coupon = self.annuity.coupon_amount * (t - ti) / (payment_time - ti)
                break
            ti = payment_time
        return self.base_strike_price + acc_coupon

    def terminal(self, stock_prices):
        """Payoff at maturity: max(adjusted strike - S, 0)."""
        strike_price = self.strike(self.T)
        return np.maximum(strike_price - stock_prices, 0)
    def transient(self, time, stock_prices, values):
        """Returns the values without changing, as the annuity doesn't change during non-terminal times."""
        return values


class ConvertibleBondCallOption(Payoff):
    """Customized call option where the strike price is adjusted for outstanding coupon."""

    def __init__(self, maturity, base_strike_price, annuity):
        super().__init__(maturity)
        self.T = maturity
        self.base_strike_price = base_strike_price
        self.annuity = annuity
    def coupon(self, time):
        """Put options usually don't have coupon payments, return 0."""
        return 0.0
    def strike(self, t):
        """Adjusts the strike price based on accrued coupons."""
        ti = 0
        acc_coupon = 0
        for payment_time in self.annuity.times:
            if payment_time > t:
                acc_coupon = self.annuity.coupon_amount * (t - ti) / (payment_time - ti)
                break
            ti = payment_time
        return self.base_strike_price + acc_coupon

    def terminal(self, stock_prices):
        """Payoff at maturity: max(S - adjusted strike, 0)."""
        strike_price = self.strike(self.T)
        return np.maximum(stock_prices - strike_price, 0)
    
    def transient(self, time, stock_prices, values):
        """Returns the values without changing, as the annuity doesn't change during non-terminal times."""
        return values
