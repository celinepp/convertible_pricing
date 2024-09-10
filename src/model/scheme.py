import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

class FiniteDifferenceScheme:
    """
    Base class for finite difference schemes.
    """

    def __init__(self, stock_prices):
        self.stock_prices = stock_prices

    def __call__(self, time_index, payoff_values, default_values, coupon_payment, transient_function):
        implicit_values = self.scheme(payoff_values, default_values)
        return transient_function(time_index, implicit_values, self.stock_prices) + coupon_payment, implicit_values

    def scheme(self, payoff_values, default_values):
        """Discount portfolio value back one period."""
        raise NotImplementedError("Subclasses should implement this method.")


class ExplicitScheme(FiniteDifferenceScheme):
    """
    Explicit finite difference scheme.
    """

    def __init__(self, stochastic_process, time_step, stock_step, stock_prices, **kwargs):
        super().__init__(stock_prices)
        a, b, c, d = stochastic_process.get_fde_params(time_step, stock_step, stock_prices, "explicit", **kwargs)
        self.L = sparse.dia_matrix(([a, 1 + b, c], [-1, 0, 1]), shape=stock_prices.shape * 2)
        self.d = d

    def scheme(self, payoff_values, default_values):
        return self.L.dot(payoff_values) + self.d * default_values


class ImplicitScheme(FiniteDifferenceScheme):
    """
    Implicit finite difference scheme.
    """

    def __init__(self, stochastic_process, time_step, stock_step, stock_prices, **kwargs):
        super().__init__(stock_prices)
        a, b, c, d = stochastic_process.get_fde_params(time_step, stock_step, stock_prices, "implicit", **kwargs)
        self.L = sparse.dia_matrix(([-a, 1 - b, -c], [-1, 0, 1]), shape=stock_prices.shape * 2).tocsr()
        self.d = d

    def scheme(self, payoff_values, default_values):
        return linalg.spsolve(self.L, payoff_values + self.d * default_values)
