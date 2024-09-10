import numpy as np
from model.process import StochasticProcess
from payoff.customized_payoff import ConvertibleBondAnnuity, ConvertibleBondPutOption, ConvertibleBondCallOption
from payoff.combined_payoff import PayoffCombination

# Time to maturity: 5 years
maturity_time = 5

# Stock price processes with different default scenarios
stock_process_total_default = StochasticProcess(risk_free_rate=0.05, volatility=0.2, hazard_rate=0.02, jump_magnitude=1)
stock_process_typical_default = StochasticProcess(risk_free_rate=0.05, volatility=0.2, hazard_rate=0.02, jump_magnitude=0.3)
stock_process_partial_default = StochasticProcess(risk_free_rate=0.05, volatility=0.2, hazard_rate=0.02, jump_magnitude=0)
stock_process_no_default = StochasticProcess(risk_free_rate=0.05, volatility=0.2)

# Bond parameters: Nominal value = 100, Semi-annual coupon = 4
annuity = ConvertibleBondAnnuity(maturity_time, np.arange(0.5, maturity_time + 0.5, 0.5), coupon_amount=4, nominal_value=100)

# American put option: Strike = 105, Time = 3 years
put_option = ConvertibleBondPutOption(maturity_time, base_strike_price=105, annuity=annuity)

# American call option: Strike = 110, Time = [2, 5] years
call_option = ConvertibleBondCallOption(maturity_time, base_strike_price=110, annuity=annuity)

# Convertible bond payoff combining the bond and equity components
convertible_bond_payoff = PayoffCombination([annuity, put_option, call_option])
