
import illiquidity_functions as iqf
import datetime as dt
import polars as pl
import sf_quant.optimizer as sfo
import os
from pathlib import Path

# Hypothesis: Illiquid stocks demand higher returns because they are more difficult to transact or have higher transaction costs. 
# Essentially selling insurance illiquid stocks in down periods perform poorly and positions are hard to exit. 

# Illiquidity should also be highly correlated with size. We construct a size signal and also construct FF6 regressions. 

if __name__ == "__main__":

    start = dt.date(1995, 6, 27)
    # start = dt.date(2023, 1, 1)
    end = dt.date(2025, 9, 15)

    # columns to pull from Barra file
    columns = [
        'date',
        'barrid',
        'ticker',
        'price',
        'return',
        'market_cap',
        'bid_ask_spread',
        'daily_volume',
        'predicted_beta',
        'specific_return',
        'specific_risk',
        'yield', 
        'iso_country_code', 
        'rootid', 
        'issuerid'
    ]

    # select whether we only want to consider Russell 3000 stocks
    russell = True

    # select number of bins
    # ie quintile vs decile sorts
    num_bins = 5

    # select filter parameters
    lag = True
    price_filter = 5
    market_cap_filter = None

    # mvo parameters
    unit_beta_constraints = [sfo.constraints.FullInvestment(), sfo.constraints.LongOnly(), sfo.constraints.NoBuyingOnMargin(), sfo.constraints.UnitBeta()]
    zero_beta_constraints = [sfo.constraints.ZeroBeta()]
    gamma = 400


    # pull Barra data
    data = iqf.get_barra_data(start, end, columns, russell)
    
    #construct signals and filter data
    # size = iqf.compute_size(data)
    data = iqf.compute_idio_vol(data)
    data = iqf.compute_price_impact_illiquidity(data)
    data = iqf.compute_cost_illiquidity(data)
    signals = ['idio_vol', 'price_impact', 'cost']

    # data = iqf.compute_idio_vol_rank(data)
    # data = iqf.compute_price_impact_illiquidity_rank(data)
    # data = iqf.compute_cost_illiquidity_rank(data)
    # signals = ['idio_vol_rank', 'price_impact_rank', 'cost_rank']


    # data = iqf.compute_barra_reversal(data)
    # data = iqf.compute_volume_adjusted_barra_reversal(data)
    # data = iqf.compute_cost_adjusted_barra_reversal(data)
    # signals = ['barra_rev', 'volume_barra_rev', 'cost_barra_rev']


    signals_str = '_'.join(signals)

    for signal in signals:
        data = iqf.compute_alphas(data, signal)

    filtered_data = iqf.filter_prices(data, signals[0], lag=lag, price_filter=price_filter, market_cap_filter=market_cap_filter)
    folder = Path("/home/bwaits/Research/Waits-Research/labs/alphas/")
    os.makedirs(folder, exist_ok=True)
    filtered_data.write_parquet(folder / f"{signals_str}.parquet")


