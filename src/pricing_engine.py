from model.binomial_model import BinomialTreeModel
from model.fde_model import FDEModel
from model.scheme import ExplicitScheme, ImplicitScheme
from bond_parameters import convertible_bond_payoff, stock_process_typical_default

class PricingEngine:
    """
    Pricing engine to price convertible bonds using different models.
    """

    def __init__(self, model, payoff, stock_process):
        self.model = model
        self.payoff = payoff
        self.stock_process = stock_process

    def price(self, initial_stock_price):
        """Calculates the price of the convertible bond."""
        return self.model.price(initial_stock_price, self.stock_process)


if __name__ == "__main__":
    # Binomial model pricing
    binomial_model = BinomialTreeModel(num_periods=100, time_increment=5 / 100, payoff_function=convertible_bond_payoff)
    pricing_engine = PricingEngine(binomial_model, convertible_bond_payoff, stock_process_typical_default)
    price = pricing_engine.price(100)
    print(f"Convertible Bond Price (Binomial Model): {price}")

    # Finite Difference model pricing
    fde_model = FDEModel(num_time_steps=100, payoff_function=convertible_bond_payoff)
    fde_price = fde_model.price(50, 150, 100, ExplicitScheme, stock_process_typical_default)
    print(f"Convertible Bond Price (FDE Model - Explicit Scheme): {fde_price}")
