# Backtesting
This is a backtesting program for purely technical based trading strategies. It works by taking a range of inputs as parameters of a particular strategy and creating a list of permutations for that strategy. Each permutation of the strategy is tested and a list of evaluation metrics are created and stored. For simplicity's sake, strategies are only tested on a handful of assets with the data for each asset spanning two months of hourly candle data. All entry and exit values are based on closing price. All gains and losses are expressed as units of a set per trade risk (R = |entry price - stop loss price|)
# Basic Outline
	Select market (currently either crypto or forex)
	select the assets you want to test for selected market
	Select the timeframe to test
	Select strategies to run on x data on y timeframe
	pull the latest data* (currently just using historical data)

	-loop through markets
	-	loop through assets
	-		loop through timeframes
	-			get appropriate data and make neccessary adjustments
	-			loop through strategies
	-				create a database table for asset-timeframe-strategy to store strategy results if one doesnt already exist
	-					get permutations list
	-					loop through permuations list
	-						run strategy for each permutation
	-						get eval metrics
	-						store eval metrics

# Evaluation Metrics
    total_realized_R: total profit in terms of units of risk *** see notes on R
    strike_rate: number of winners/total trades
    num_trades: number of trades
    avg_hold_time: average holding time of trades in seconds
    expectancy: the expected return of a strategy per trade taken measured in R, (avg_winner * win_strike_rate) - abs(avg_loser * loss_strike_rate)
    avg_winner: average R gain per trade
    avg_loser: average R loss per trade (if < -1 then strategy has considerable slippage)
    largest_winner: largest single winner in units of risk
    largest_loser: largest single loser in units of risk
    max_drawdown: longest consecutive string of losers
    max_drawup: longest consecutive string of winners
    winners_std_MAE: standard deviation of MAE for all winning trades *** see MAE/MFE notes
    winners_std_MFE: standard deviation of MFE for all winning trades *** see MAE/MFE notes
    losers_std_MAE: standard deviation of MAE for all losing trades *** see MAE/MFE notes
    losers_std_MFE: standard deviation of MFE for all losing trades *** see MAE/MFE notes
    kelly_criterion: bet sizing metric *** see kelly criterion notes
    equity_curve_regression_slope: slope of the regression line of the equity curve *** see equity curve regression
    equity_curve_regression_std_error: average of all residuals *** see equity curve regression
    trades_df: dataframe of all the raw trade data (used to build equity curve)
	
## R Multiples
R multiples are set, per trade, units of risk (R) measured in multiples. 
i.e if the entry is 1.00, stop loss is 0.90, and profit exit is 1.20, R = |entry - stop| => |1.00 - 0.90| = 0.10,
since the target price is 0.20 from the entry then (0.20 / R) = 2, a 2R profit.
Every metric that can deal with price in the evaluation metrics uses R as a substitute to normalize the results
	
## MAE/MFE
maximum adverse excurison (MAE) - the most a trade goes against you (measured as multiples of R) before being closed out
maximum favorable excurison (MFE) - the most a trade goes in your favor (measured as multiples of R) before being closed out

For winning trades, the MAE is how far a trade went against you before hitting the profit target
For losing trades, the MFE is how far a trade went in profit before hitting the stop loss

For a series of trades the standard deviation of winning MAE and losing MFE is calculated to give insight into aggregate trade behavior. 
For example, since the confidence interval of observations within 1 standard deviation is 68%, if the standard deviation of MFE for losing trades is 0.5R
then it can reasonably be assumed that if an open trade gets above a 0.5R profit, it stands a much better chance of hitting its target. Conversely, if the standard
deviation of MAE for winning trades is -0.3R and an open trade drops below a 0.3R loss, then it can be assumed the trades chances of becoming a loss are increased.

Therefore since the outcome of an open trade can be infered from whether or not price hits specific R levels, adjustments to stops or targets can be made to increase expectancy.

## Kelly Criterion
Betting optimization formula meant to maximize performance.

	kelly = W - ((1 - W) / (P / L))	
	
	W = win rate 
	(1 - W) = loss rate
	P / L = total profit to total loss
	
The higher the better. Kelly can be interpretted as the percentage of total capital to be staked on the next trade to maximize performance

## Equity Curve Regression
Slope
Takes the equity curve of the strategy and fits a regression line to it, the slope of that line determines if its profitable and the degree of profitability.
A larger positive slope means the strategy is more profitable

Standard Error
Takes the residual errors of the regression line and averages them. A larger average standard error indicates a higher level of volatility/cyclicality in returns
whereas a smaller average standard error indicates a better fitting regression line and less volatility/cyclicality in returns
