
import illiquidity_functions as iqf
import datetime as dt
import polars as pl
import sf_quant.optimizer as sfo

# Hypothesis: Illiquid stocks demand higher returns because they are more difficult to transact or have higher transaction costs. 
# Essentially selling insurance illiquid stocks in down periods perform poorly and positions are hard to exit. 

# Illiquidity should also be highly correlated with size. We construct a size signal and also construct FF6 regressions. 

if __name__ == "__main__":

    start = dt.date(1996, 1, 1)
    # start = dt.date(2023, 1, 1)
    end = dt.date(2024, 12, 31)

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
    num_bins = 10

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
    
    # construct signals and filter data
    # size = iqf.compute_size(data)
    # idio_vol = iqf.compute_idio_vol(data)
    # price_impact_illiquidity = iqf.compute_price_impact_illiquidity(data)
    # cost_illiquidity = iqf.compute_cost_illiquidity(data)

    volume_z = iqf.compute_volume_z(data)

    # size = iqf.compute_alphas(size, 'size')
    # idio_vol = iqf.compute_alphas(idio_vol, 'idio_vol')
    # price_impact_illiquidity = iqf.compute_alphas(price_impact_illiquidity, 'price_impact')
    # cost_illiquidity = iqf.compute_alphas(cost_illiquidity, 'cost')

    # filtered_size = iqf.filter_prices(size, 'size', lag=lag, price_filter=price_filter, market_cap_filter=market_cap_filter)
    # filtered_idio_vol = iqf.filter_prices(idio_vol, 'idio_vol', True, price_filter, market_cap_filter)
    # filtered_price_impact_illiquidity = iqf.filter_prices(price_impact_illiquidity, 'price_impact', lag=lag, price_filter=price_filter, market_cap_filter=market_cap_filter)
    # filtered_cost_illiquidity = iqf.filter_prices(cost_illiquidity, 'cost', lag=lag, price_filter=price_filter, market_cap_filter=market_cap_filter)

    filtered_volume_z = iqf.filter_prices(volume_z, 'volume_z', lag=lag, price_filter=price_filter, market_cap_filter=market_cap_filter)

    # size_weights = iqf.compute_mvo_backtest(filtered_size, zero_beta_constraints, gamma)
    # idio_vol_weights = iqf.compute_mvo_backtest(filtered_idio_vol, zero_beta_constraints, gamma)
    # price_impact_illiquidity_weights = iqf.compute_mvo_backtest(filtered_price_impact_illiquidity, zero_beta_constraints, gamma)
    # cost_illiquidity_weights = iqf.compute_mvo_backtest(filtered_cost_illiquidity, zero_beta_constraints, gamma)

    # bin_data_size = iqf.compute_bins(filtered_size, 'size', num_bins,)
    # bin_data_idio_vol = iqf.compute_bins(filtered_idio_vol, 'idio_vol', num_bins)
    # bin_data_price_impact_illiquidity = iqf.compute_bins(filtered_price_impact_illiquidity, 'price_impact', num_bins,)
    # bin_data_cost_illiquidity = iqf.compute_bins(filtered_cost_illiquidity, 'cost', num_bins,)

    bin_data_volume_z = iqf.compute_bins(filtered_volume_z, 'volume_z', num_bins,)

    # size_portfolios = iqf.construct_equal_weight_portfolios(bin_data_size, 'size', num_bins)
    # idio_vol_portfolios = iqf.construct_equal_weight_portfolios(bin_data_idio_vol, 'idio_vol', num_bins)
    # price_impact_illiquidity_portfolios = iqf.construct_equal_weight_portfolios(bin_data_price_impact_illiquidity, 'price_impact', num_bins)
    # cost_illiquidity_portfolios = iqf.construct_equal_weight_portfolios(bin_data_cost_illiquidity, 'cost', num_bins)

    volume_z_portfolios = iqf.construct_equal_weight_portfolios(bin_data_volume_z, 'volume_z', num_bins)

    # size_portfolio_returns = iqf.compute_cumulative_log_returns(size_portfolios, 'size', num_bins)
    # idio_vol_portfolio_returns = iqf.compute_cumulative_log_returns(idio_vol_portfolios, 'idio_vol', num_bins)
    # price_impact_illiquidity_portfolio_returns = iqf.compute_cumulative_log_returns(price_impact_illiquidity_portfolios, 'price_impact', num_bins)
    # cost_illiquidity_portfolio_returns = iqf.compute_cumulative_log_returns(cost_illiquidity_portfolios, 'cost', num_bins)

    volume_z_portfolio_returns = iqf.compute_cumulative_log_returns(volume_z_portfolios, 'volume_z', num_bins)

    # plot binned portfolios and spread
    # iqf.plot_portfolio_log_returns(size_portfolios, 'size', num_bins) 
    # iqf.plot_portfolio_log_returns(idio_vol_portfolios, 'idio_vol', num_bins)
    # iqf.plot_portfolio_log_returns(price_impact_illiquidity_portfolios, 'price_impact', num_bins)
    # iqf.plot_portfolio_log_returns(cost_illiquidity_portfolios, 'cost', num_bins)

    iqf.plot_portfolio_log_returns(volume_z_portfolios, 'volume_z', num_bins)

    # display portfolio metrics
    # iqf.calculate_summary_stats(size_portfolio_returns, 'size')
    # iqf.calculate_summary_stats(idio_vol_portfolio_returns, 'idio_vol')
    # iqf.calculate_summary_stats(price_impact_illiquidity_portfolio_returns, 'price_impact')
    # iqf.calculate_summary_stats(cost_illiquidity_portfolio_returns, 'cost')

    iqf.calculate_summary_stats(volume_z_portfolio_returns, 'volume_z')


    # iqf.calculate_ff_regression(size_portfolios, 'size')
    # iqf.calculate_ff_regression(idio_vol_portfolios, 'idio_vol')
    # iqf.calculate_ff_regression(price_impact_illiquidity_portfolios, 'price_impact')
    # iqf.calculate_ff_regression(cost_illiquidity_portfolios, 'cost')

    iqf.calculate_ff_regression(volume_z_portfolios, 'volume_z')


    # MVO portfolio results
    # iqf.construct_mvo_results(size_weights, 'size', 'zero_beta')
    # iqf.construct_mvo_results(idio_vol_weights, 'idio_vol', 'zero_beta')
    # iqf.construct_mvo_results(price_impact_illiquidity_weights, 'price_impact', 'zero_beta')
    # iqf.construct_mvo_results(cost_illiquidity_weights, 'cost', 'zero_beta')


