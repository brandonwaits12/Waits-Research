
import illiquidity_functions as iqf
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

    # signals = ['price_impact', 'cost']
    signals = ['volume_z']
    # constraint_types = ['zero_beta']
    for signal in signals:
        
        folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
        backtest_results = pl.read_parquet(folder / f"{signal}_decile_backtest.parquet")
        print(f"{signal} Binned Portfolio Backtest")
        print(backtest_results)
        print()

        print(f"{signal} Factor Regression")
        regression_results = pl.read_parquet(folder / f"{signal}_factor_regression.parquet")
        print(regression_results)
        print()

        # print(f"{signal} MVO")
        # MVO_results = pl.read_parquet(folder / f"{signal}_{constraint_types[0]}_mvo_backtest.parquet")
        # print(MVO_results)
        # print()
        







