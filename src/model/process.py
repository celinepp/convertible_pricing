import numpy as np

class StochasticProcess:
    """
    Models a stochastic process with both Wiener drift and Poisson jump processes
    for simulating stock price movements and default events.

    Attributes:
        risk_free_rate (float): The risk-free interest rate.
        volatility (float): The volatility of the stock price.
        hazard_rate (float or callable): The hazard rate for the Poisson process.
        jump_magnitude (float): The magnitude of the drop in stock price upon default.
        cap_hazard_rate (bool): Whether to cap the hazard rate based on volatility.
    """

    def __init__(self, risk_free_rate, volatility, hazard_rate=0, jump_magnitude=1, cap_hazard_rate=False):
        if volatility <= 0:
            raise ValueError("Volatility must be positive.")
        if hazard_rate < 0:
            raise ValueError("Hazard rate must be non-negative.")

        self.risk_free_rate = np.double(risk_free_rate)
        self.volatility = np.double(volatility)
        self.hazard_rate = hazard_rate if callable(hazard_rate) else np.double(hazard_rate)
        self.jump_magnitude = np.double(jump_magnitude)
        self.cap_hazard_rate = cap_hazard_rate

    def get_binomial_params(self, time_step, stock_price=None):
        """
        Returns parameters for the binomial model.

        Args:
            time_step (float): The time increment for each step in the binomial model.
            stock_price (float or None): The current stock price, if required for hazard rate calculation.

        Returns:
            tuple: Contains the up factor, down factor, loss factor, up probability, down probability, and jump probability.
        """
        if time_step <= 0:
            raise ValueError("Time step must be positive.")

        # Up and down multipliers
        up_factor = np.exp(self.volatility * np.sqrt(time_step))
        down_factor = 1 / up_factor
        loss_factor = 1 - self.jump_magnitude

        # Calculate hazard rate
        hazard_rate = self._get_hazard_rate(stock_price)
        hazard_rate_limit = (np.log(up_factor - loss_factor) - np.log(np.exp(self.risk_free_rate * time_step) - loss_factor))

        if self.cap_hazard_rate:
            hazard_rate = np.minimum(hazard_rate, hazard_rate_limit / time_step)
        elif hazard_rate * time_step > hazard_rate_limit:
            raise ValueError("Time step too large for the given hazard rate.")

        # Probabilities of up, down, and default
        jump_probability = 1 - np.exp(-hazard_rate * time_step)
        up_probability = (np.exp(self.risk_free_rate * time_step) - down_factor * (1 - jump_probability) - loss_factor * jump_probability) / (up_factor - down_factor)
        down_probability = 1 - up_probability - jump_probability

        return up_factor, down_factor, loss_factor, up_probability, down_probability, jump_probability

    def get_fde_params(self, time_step, stock_step, stock_prices, scheme, boundary="diffequal", expfit=False):
        """
        Returns parameters for the finite difference scheme.

        Args:
            time_step (float): The time increment for each step.
            stock_step (float): The stock price increment.
            stock_prices (numpy array): Array of stock prices.
            scheme (str): The finite difference scheme to use.
            boundary (str): Boundary conditions for the FDE model.
            expfit (bool): Whether to use exponential fitting.

        Returns:
            tuple: Contains the finite difference coefficients.
        """
        if time_step <= 0:
            raise ValueError("Time step must be positive.")
        if stock_step <= 0:
            raise ValueError("Stock step must be positive.")
        if (stock_prices < 0).any():
            raise ValueError("Stock prices must be non-negative.")

        hazard_rate = self._get_hazard_rate(stock_prices)
        adjusted_rate = (self.risk_free_rate + hazard_rate) * time_step
        drift = time_step * (self.risk_free_rate + hazard_rate * self.jump_magnitude) * stock_prices / stock_step / 2

        if expfit:
            x = (self.risk_free_rate + hazard_rate * self.jump_magnitude) * stock_step / self.volatility ** 2 / stock_prices
            coth = 1. / np.tanh(x)
            diffusion = time_step * coth * (self.risk_free_rate + hazard_rate * self.jump_magnitude) * stock_prices / stock_step / 2
        else:
            diffusion = time_step * self.volatility ** 2 * stock_prices ** 2 / stock_step ** 2 / 2

        a = diffusion[1:] - drift[1:]
        b = -adjusted_rate - 2 * diffusion
        c = diffusion[:-1] + drift[:-1]
        d = hazard_rate * time_step

        if boundary == "equal":
            b[0] += diffusion[0] - drift[0]
            b[-1] += diffusion[-1] + drift[-1]
        elif boundary == "diffequal":
            b[0] += 2 * (diffusion[0] - drift[0])
            c[0] -= diffusion[0] - drift[0]
            b[-1] += 2 * (diffusion[-1] + drift[-1])
            a[-1] -= diffusion[-1] + drift[-1]
        elif boundary != "ignore":
            raise ValueError(f"Unknown boundary type: {boundary}")

        return np.append(a, diffusion[0] - drift[0]), b, np.append(diffusion[-1] + drift[-1], c), d

    def _get_hazard_rate(self, stock_price):
        """
        Returns the hazard rate based on the stock price.

        Args:
            stock_price (float or numpy array): The current stock price.

        Returns:
            float or numpy array: The hazard rate.
        """
        if callable(self.hazard_rate):
            return self.hazard_rate(stock_price)
        return self.hazard_rate
