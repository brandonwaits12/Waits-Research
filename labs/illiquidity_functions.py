import sf_quant.data as sfd
import sf_quant.optimizer as sfo
import sf_quant.backtester as sfb
import sf_quant.performance as sfp
import polars as pl
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
import os
from pathlib import Path
import pandas as pd


def get_barra_data(start: dt.date, end: dt.date, columns: list[str], russell: bool):

    # data loader
    data = sfd.load_assets(
        start=start,
        end=end,
        in_universe=russell, # bool to indicate if we are only using Russell 3000 stocks
        columns=columns
    )

    # convert returns to decimal space
    data = (
        data.with_columns(
            pl.col('return').truediv(100).alias('return')))

    data = (
        data.with_columns(
            pl.col('specific_return').truediv(100).alias('specific_return')))

    data = (
        data.with_columns(
            pl.col('specific_risk').truediv(100).alias('specific_risk')
        )
    )

    data = (
        data.with_columns(
            pl.col("daily_volume").replace(0, None)
        )
    )

    if not russell:
        # filter out securities we dont want to consider if we are using stocks outside of Russell 3000
        data = data.filter(pl.col('iso_country_code').eq("USA"), pl.col('rootid').eq(pl.col('barrid')), pl.col('barrid').str.starts_with('US')) #pl.col('price').ge(5))

    return data


def compute_size(data: pl.DataFrame) -> pl.DataFrame:

    # the inverse of the market cap. rolling average to smooth. 
    # smaller stocks should yield higher returns 
    size = (
        data.with_columns(
            pl.col("market_cap").mul(-1).rolling_mean(window_size=252, min_samples = 252)
            .shift(2)
            .over('barrid')
            .alias('size')
        )
        .sort(['barrid', 'date'])
    )
    return size


def compute_price_impact_illiquidity(data: pl.DataFrame) -> pl.DataFrame:

    # variation of Amihud illiquidity metric. 
    # measure how much movement in stock price for each unit of dollar traded volume
    price_impact = (
        data.with_columns(
            (pl.col("return").abs().mul(1).truediv(pl.col('daily_volume').mul(pl.col('price')))).rolling_mean(window_size=22, min_samples = 22)
            .shift(2) 
            .over('barrid')
            .alias('price_impact')
        )
        .sort(['barrid', 'date'])
    )

    # price_impact = (
    #     data.with_columns(
    #         (pl.col("return").abs().mul(1).truediv((pl.col('daily_volume').mul('price')).truediv('market_cap'))).rolling_mean(window_size=22, min_samples = 22)
    #         .shift(2) 
    #         .over('barrid')
    #         .alias('price_impact')
    #     )
    #     .sort(['barrid', 'date'])
    # )

    price_impact = (
        data
        .sort(["barrid", "date"])
        .with_columns(
            shrout = pl.col("market_cap") / pl.col("price")
        )
        .with_columns(
            shrout_delta = (pl.col("shrout") / pl.col("shrout").shift(1) - 1.0).over("barrid")
        )
        .with_columns(
            price_impact = (-pl.col("shrout_delta")).shift(2).over("barrid")
        )
    )
        

    # Use this constrution if you dont want to drop 0 volume days
    # price_impact = (
    #     data.with_columns(
    #         (pl.col("return").abs().mul(pl.col('daily_volume').mul(pl.col('price')))).rolling_mean(window_size=22, min_samples = 22)
    #         .mul(1)
    #         .shift(2) 
    #         .over('barrid')
    #         .alias('price_impact')
    #     )
    #     .sort(['barrid', 'date'])
    # )
    return price_impact

def compute_price_impact_illiquidity_rank(data: pl.DataFrame) -> pl.DataFrame:
        # variation of Amihud illiquidity metric. 
    # measure how much movement in stock price for each unit of dollar traded volume
    price_impact = (
        data.with_columns(
            (pl.col("return").abs().mul(1).truediv(pl.col('daily_volume').mul(pl.col('price')))).rolling_mean(window_size=22, min_samples = 22)
            .mul(-1)
            .shift(2) 
            .over('barrid')
            .alias('price_impact')
        ).with_columns(
        pl.col('price_impact')
        .rank('ordinal')
        .over('date')
        .alias('price_impact_rank')
        )
        .sort(['barrid', 'date'])
    )
    return price_impact



def compute_cost_illiquidity(data: pl.DataFrame) -> pl.DataFrame:

    # the other element of illiquidity besides price impact is transaction cost, which can be measured using the b/a spread
    cost = (
        data.with_columns(
            (pl.col("bid_ask_spread").truediv(pl.col('price'))).pow(1).rolling_mean(window_size=252, min_samples = 252)
            .truediv(pl.col("return").rolling_std(window_size=44, min_samples = 44))
            .shift(2)
            .over('barrid')
            .alias('cost')
        )
        .sort(['barrid', 'date'])
    )
    return cost

def compute_cost_illiquidity_rank(data: pl.DataFrame) -> pl.DataFrame:

    # the other element of illiquidity besides price impact is transaction cost, which can be measured using the b/a spread
    cost = (
        data.with_columns(
            (pl.col("bid_ask_spread").truediv(pl.col('price'))).pow(1).rolling_mean(window_size=252, min_samples = 252)
            .truediv(pl.col("return").rolling_std(window_size=44, min_samples = 44))
            #reversed
            .mul(-1)
            .shift(2)
            .over('barrid')
            .alias('cost')
        ).with_columns(
        pl.col('cost')
        .rank('ordinal')
        .over('date')
        .alias('cost_rank')
        )
        .sort(['barrid', 'date'])
    )
    return cost



def compute_idio_mom(data: pl.DataFrame) -> pl.DataFrame:

    # momentum enhance: factor neutral vol scaled momentum
    idio_mom = (
        data.with_columns(
            (pl.col("specific_return").rolling_mean(window_size=252, min_samples = 252))
            .truediv(pl.col("specific_risk").rolling_mean(window_size=44, min_samples = 44))
            .shift(22)
            .over('barrid')
            .alias('idio_mom')
        )
        .sort(['barrid', 'date'])
    )
    return idio_mom

def compute_idio_vol_rank(data: pl.DataFrame) -> pl.DataFrame:

    # long low idio vol stocks, short high idio vol stocks
    idio_vol = (
        data.sort(['barrid', 'date']).with_columns(
            (pl.col("specific_risk").rolling_mean(window_size=252, min_samples = 252))
            # .mul(-1)
            .shift(2)
            .over('barrid')
            .alias('idio_vol')
        ).sort(['barrid', 'date']).with_columns(
        pl.col('idio_vol')
        .rank('ordinal')
        .over('date')
        .alias('idio_vol_rank')
        )
        .sort(['barrid', 'date'])
    )
    return idio_vol

def compute_idio_vol(data: pl.DataFrame) -> pl.DataFrame:

    # long low idio vol stocks, short high idio vol stocks
    idio_vol = (
        data.with_columns(
            (pl.col("specific_risk").rolling_mean(window_size=252, min_samples = 252))
            .mul(-1)
            .shift(2)
            .over('barrid')
            .alias('idio_vol')
        )
        .sort(['barrid', 'date'])
    )
    return idio_vol

def compute_barra_reversal(data: pl.DataFrame) -> pl.DataFrame:

    barra_reversal = (
        data.with_columns(
            (pl.col("specific_return").ewm_mean(span=5, min_samples = 5))
            .mul(-1)
            .shift(2)
            .over('barrid')
            .alias('barra_rev')
        )
        .sort(['barrid', 'date'])
    )
    return barra_reversal

def compute_volume_adjusted_barra_reversal(data: pl.DataFrame) -> pl.DataFrame:

    volume_adjusted_barra_reversal = (
        data
        .sort(["barrid", "date"])
        .with_columns(
            dollar_volume = pl.col("daily_volume").mul(pl.col("price")))
        .with_columns(
            volume_score = (
                (pl.col("dollar_volume").sub(pl.col("dollar_volume").rolling_mean(window_size=88, min_periods=88)))
                .truediv(pl.col("dollar_volume").rolling_std(window_size=88, min_periods=88))
            )
        )
        .with_columns(
            volume_barra_rev = (
                pl.when(pl.col("volume_score") > 1.5)
                .then(0.0)
                .otherwise(pl.col("specific_return").ewm_mean(span=5, min_periods=5))
                .mul(-1)
                .shift(2)
                .over("barrid")
            )
        )
    )
    return volume_adjusted_barra_reversal

def compute_cost_adjusted_barra_reversal(data: pl.DataFrame) -> pl.DataFrame:

    cost_adjusted_barra_reversal = (
        data
        .sort(["barrid", "date"])
        .with_columns(
            price_adj_spread = pl.col("bid_ask_spread").truediv(pl.col("price")))
        .with_columns(
            spread_score = (
                (pl.col("price_adj_spread").sub(pl.col("price_adj_spread").rolling_mean(window_size=88, min_periods=88)))
                .truediv(pl.col("price_adj_spread").rolling_std(window_size=88, min_periods=88))
            )
        )
        .with_columns(
            cost_barra_rev = (
                pl.when(pl.col("spread_score") > 1.5)
                .then(0.0)
                .otherwise(pl.col("specific_return").ewm_mean(span=5, min_periods=5))
                .mul(-1)
                .shift(2)
                .over("barrid")
            )
        )
    )
    return cost_adjusted_barra_reversal

def compute_volume_z(data: pl.DataFrame) -> pl.DataFrame:
    # volume_z = (
    #     data
    #     .sort(["barrid", "date"])
    #     .with_columns(
    #         dollar_volume = pl.col("daily_volume").mul(pl.col("price")))
    #     .with_columns(
    #         volume_z = (
    #             pl.col('return').ewm_mean(span=22, min_samples=22)
    #             .mul(-1)
    #             .mul(pl.col("dollar_volume").sub(pl.col("dollar_volume").rolling_mean(window_size=252, min_periods=252)))
    #             .truediv(pl.col("dollar_volume").rolling_std(window_size=252, min_periods=252))
    #         )
    #         .shift(2)
    #         .over("barrid")
    #     )
    #     .sort(["barrid", "date"])
    # )
    volume_z = (
        data
        .sort(["barrid", "date"])
        .with_columns(
            dollar_volume = pl.col("daily_volume").mul(pl.col("price")))
        .with_columns(
            volume_z = (
                pl.col("dollar_volume").sub(pl.col("dollar_volume").rolling_mean(window_size=252, min_periods=252))
                .truediv(pl.col("dollar_volume").rolling_std(window_size=252, min_periods=252))
            )
            .shift(2)
            .over("barrid")
        )
        .sort(["barrid", "date"])
    )
    return volume_z


def compute_alphas(data: pl.DataFrame, signal: str) -> pl.DataFrame:

    # alphas = (data.sort(["barrid", "date"])
    #     .with_columns(
    #         m = pl.col(signal).mean().over("date"),
    #         s = pl.col(signal).std().over("date"),
    #     )
    #     .with_columns(
    #         score = pl.when(pl.col("s") > 0)
    #                   .then((pl.col(signal) - pl.col("m")) / pl.col("s"))
    #                   .otherwise(0.0)
    #                   .clip(-2.0, 2.0)
    #     )
    #     .with_columns(
    #         pl.col('score').mul(0.05).mul('specific_risk').alias(f"{signal}_alpha")
    #     )
    # )

    alphas = (data.with_columns(
            ((pl.col(f"{signal}").sub(pl.col(f"{signal}").mean().over("date"))).truediv(pl.col(f"{signal}").std().over("date"))).alias("score")
        )
        .with_columns(
            (pl.col("score").mul(pl.col("specific_risk"))).alias(f"{signal}_alpha")
        )
        .sort(['barrid', 'date'])
    )
    
    return alphas


def filter_prices(data: pl.DataFrame, signal: pl.DataFrame, lag: bool, price_filter: pl.Int64 | None = None, market_cap_filter: pl.Int64 | None = None) -> pl.DataFrame:

    # filter out securities which will cause errors in portfolio calculuations such as null/NaN values
    # filter out low-price securities (penny stocks) as they are often non tradeable or skew results. 

    data = data.sort(["barrid", "date"]).with_columns(
            pl.col("price")
            .shift(1)
            .over("barrid")
            .alias("price_lag")
            )
    
    data = data.sort(["barrid", "date"]).with_columns(
            pl.col("market_cap")
            .shift(1)
            .over("barrid")
            .alias("market_cap_lag")
            )
    

    # filter out no signal data
    data = data.filter(pl.col(signal).is_not_null(), pl.col(signal).is_not_nan())

    # filter out no alpha data if alpha column exists
    if 'alpha' in data.columns:
        data = data.filter(
            pl.col('alpha').is_not_null(), 
            pl.col('alpha').is_not_nan()
        )

    # filter based on parameters
    if lag:
        if price_filter:
            data = data.filter(pl.col("price_lag") >= price_filter)
        if market_cap_filter:
            data = data.filter(pl.col("market_cap_lag") >= market_cap_filter)

    else:
        if price_filter:
            data = data.filter(pl.col("price") >= price_filter)
        if market_cap_filter:
            data = data.filter(pl.col("market_cap") >= market_cap_filter)
    
    return data


def compute_mvo_backtest(data: pl.DataFrame, constraints, gamma) -> pl.DataFrame:

    weights = sfb.backtest_parallel(
        data=data,
        constraints=constraints,
        gamma=gamma,
        n_cpus=32
    )
    return weights


def compute_bins(data: pl.DataFrame, signal: str, num_bins: pl.Int64) -> pl.DataFrame:

    # bin the signal for decile and spread portfolio construction

    labels = [str(i) for i in range(num_bins)]
    
    bins = (
        data.with_columns(
            pl.col(signal)
            .qcut(num_bins, labels=labels, allow_duplicates=True)
            .over('date')
            .alias(f'{signal}_bin')
        )
        .sort(['barrid', 'date'])
    )
    return bins


def construct_equal_weight_portfolios(bins: pl.DataFrame, signal: str, num_bins: pl.Int64) -> pl.DataFrame:
    
    
    ew_port = (
        bins.group_by(['date', f'{signal}_bin'])
        .agg(pl.col('return').mean())
        .pivot(on=f'{signal}_bin', index='date', values='return')

        # force spread bin to use equal ex-post vol of high and low bins 
        .with_columns((pl.col(f'{num_bins-1}') - pl.col('0') * (pl.col(f'{num_bins-1}').std().truediv(pl.col('0').std())) ).alias('spread'))

        # no change to spread portfolio
        # .with_columns((pl.col(f'{num_bins-1}') - pl.col('0')).alias('spread'))

        .sort('date')
    )
    
    return ew_port


def compute_cumulative_log_returns(portfolio: pl.DataFrame, signal: str, num_bins: pl.Int64) -> pl.DataFrame:
    
    # compute cumulative return metrics for plotting and calculating statistics
    portfolio_returns = (
        portfolio
        .unpivot(
            index="date",
            on=[str(i) for i in range(num_bins)] + ["spread"],
            variable_name=f"{signal}_bin",
            value_name="return"
        )
        .with_columns([
            (pl.col("return") * 100).alias("return_pct"),
            pl.col("return").log1p().cum_sum().alias("cum_log_return")
        ])
        .sort("date")
    )
    return portfolio_returns


def plot_portfolio_log_returns(portfolio_returns: pl.DataFrame, signal: str, num_bins: pl.Int64):

    bins_labels = [str(i) for i in range(num_bins)] + ["spread"]

    # compute cumulative log returns for each portfolio column
    cumulative_portfolio_returns = (
        portfolio_returns
        .with_columns([
            pl.col(c).log1p().cum_sum().alias(c) for c in bins_labels
        ])
        .select(["date"] + bins_labels)
        .sort("date")
    )

    plt.figure(figsize=(10, 6))

    labels = [str(i) for i in range(num_bins)]
    colors = sns.color_palette('coolwarm', n_colors=len(labels))

    for color, label in zip(colors, labels):
        sns.lineplot(cumulative_portfolio_returns, x='date', y=label, label=label, color = color)

    sns.lineplot(cumulative_portfolio_returns, x='date', y='spread', color = 'black', label='Spread')

    plt.ylabel("Cumulative Log Returns")
    plt.title(f'Decile Portfolios for {signal}')
    plt.legend()
    # plt.show()

    folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder / f"{signal}_plot.png")
    
    return


def calculate_summary_stats(port_returns: pl.DataFrame, signal: str):
    stats = (
        port_returns
        .group_by(f"{signal}_bin")
        .agg([
            (pl.col("return").mean() * 252).alias("avg_return_ann"),
            (pl.col("return").std() * np.sqrt(252)).alias("vol_ann")
        ])
        .with_columns(
            (pl.col("avg_return_ann") / pl.col("vol_ann")).alias("sharpe_ann")
        )
        .sort(f"{signal}_bin")
    )
    stats_df = stats.to_pandas()
    folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
    os.makedirs(folder, exist_ok=True)
    stats_df.to_parquet(folder / f"{signal}_decile_backtest.parquet")

    print(f'{signal} Stats')
    print(stats_df)
    print()

    return 


def calculate_ff_regression(data: pl.DataFrame, signal: str):
    factors = pl.read_csv("/home/bwaits/Research/Waits-Research/labs/ff6_daily.csv").cast({"date": pl.Date})
    port = data.clone()

    # add the factor portfolios to portfolio dataframe
    port = port.join(factors, on="date", how="left").drop_nulls()

    port = port.to_pandas() # convert to pandas to be compatible with statsmodels
    port = port.set_index("date")

    # run multivariate regression, testing the ff3 + data
    # x_variables = '~ 1 + mktrf + smb + hml + umd + cma + rmw'
    x_variables = '~ 1 + mktrf + smb + hml + umd'
    # reg = smf.ols("spread" + x_variables, data=port).fit().summary()
    # reg = reg.tables[1]

    model = smf.ols("spread" + x_variables, data=port).fit()
 
    ci = model.conf_int()
    reg_pd = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t": model.tvalues.values,
        "p": model.pvalues.values,
        "ci_lo": ci[0].values,
        "ci_hi": ci[1].values,
    })

    # ---- back to Polars for saving as Parquet ----
    reg_pl = pl.from_pandas(reg_pd.reset_index(drop=True))
    

    folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
    os.makedirs(folder, exist_ok=True)
    reg_pl.write_parquet(folder / f"{signal}_factor_regression.parquet")

    print(f'{signal} Regression Stats')
    print(reg_pl)
    print()


def construct_mvo_results(weights: pl.DataFrame, signal: str, constraint_type: str):
    folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
    os.makedirs(folder, exist_ok=True)

    returns = sfp.generate_returns_from_weights(weights=weights)

    returns.write_parquet(folder / f"{signal}_{constraint_type}_mvo_backtest_data.parquet")

    portfolio_returns = sfp.generate_returns_chart(returns = returns,
                            title=f"{signal} {constraint_type} MVO Backtest",
                            log_scale=True,
                            file_name=folder / f"{signal}_{constraint_type}_mvo_backtest.png")
    


    summary = sfp.generate_summary_table(
        returns = returns
    )
    
    summary.write_parquet(folder / f"{signal}_{constraint_type}_mvo_backtest_summary.parquet")

    print(f'{signal} MVO summary')
    print(summary)
    print()


