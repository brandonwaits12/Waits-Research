
import illiquidity_functions as iqf
import sf_quant.data as sfd
import sf_quant.optimizer as sfo
import sf_quant.backtester as sfb
import sf_quant.performance as sfp
import datetime as dt
import polars as pl
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Hypothesis: Illiquid stocks demand higher returns because they are more difficult to transact or have higher transaction costs. 
# Essentially selling insurance illiquid stocks in down periods perform poorly and positions are hard to exit. 

# Illiquidity should also be highly correlated with size. We construct a size signal and also construct FF6 regressions. 

if __name__ == "__main__":

    signals = ['idio_vol', 'price_impact', 'cost']
    # signals = ['idio_vol_rank', 'price_impact_rank', 'cost_rank']
    # signals = ['barra_rev', 'volume_barra_rev', 'cost_barra_rev']
    constraint_types = ['zero_beta']
    for signal in signals:
        
        weights_folder = Path("/home/bwaits/Research/Waits-Research/labs/weights")
        illiquidity_results_folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
        backtest_weights = pl.read_parquet(weights_folder / f"{signal}_weights_*.parquet")
        print(f"{signal} Zero Beta Backtest")

        returns = sfp.generate_returns_from_weights(weights=backtest_weights)

        returns.write_parquet(illiquidity_results_folder / f"{signal}_{constraint_types[0]}_mvo_backtest_data.parquet")

        portfolio_returns = sfp.generate_returns_chart(returns = returns,
                                title=f"{signal} {constraint_types[0]} MVO Backtest",
                                log_scale=True,
                                file_name=illiquidity_results_folder / f"{signal}_{constraint_types[0]}_mvo_backtest.png", 
                                constraint='zero_beta')
        


        summary = sfp.generate_summary_table(
            returns = returns
        )
        
        summary.write_parquet(illiquidity_results_folder / f"{signal}_{constraint_types[0]}_mvo_backtest_summary.parquet")

        print(f'{signal} MVO summary')
        print(summary)
        print()

        







