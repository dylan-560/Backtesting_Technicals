import time
import os
import settings
from os.path import exists
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import json
import math
from settings import TOTAL_MONTHS_OF_DATA,BACKTEST_MONTHS


def reorder_df(df,top_line_most_recent):
    """reorders df on date and resets index"""
    if top_line_most_recent:
        df = df.sort_values(by='Date_Time', ascending=False)
    else:
        df = df.sort_values(by='Date_Time')
    df.reset_index(inplace=True, drop=True)
    return df

def slice_backtest_ohlcv_df(df, offset_candles, start_date, end_date):
    """
    cuts candle df down to (latest month start - BACKTEST_MONTHS) - offset candles (for giving technical indicators time to set up)
    """

    def convert_to_datetime(input):
        if not isinstance(input, datetime):
            input = datetime.strptime(input, '%Y-%m-%d')
        return input

    #############################################

    # reorder ohlcv data df
    df = reorder_df(df=df, top_line_most_recent=True)

    end_date = convert_to_datetime(input=end_date)
    start_date = convert_to_datetime(input=start_date)

    # add in offset candles
    temp_df = df.loc[(df['Date_Time'] > start_date) & (df['Date_Time'] <= end_date)]
    start_index = temp_df.index[-1]
    start_index += (offset_candles + 2)  # plus 2 to be safe

    end_index = temp_df.index[0]

    df = df[end_index: start_index]

    return df

def get_last_x_from_start_of_month(months=TOTAL_MONTHS_OF_DATA):
    """
    gets a start date (first of the month for current month)
    and the first of the month x months TOTAL_MONTHS_OF_DATA back
    """
    curr_month_start = datetime.today().replace(day=1)

    last_x_months_start = curr_month_start
    for i in range(0, months):
        last_x_months_start -= timedelta(days=1)
        last_x_months_start = last_x_months_start.replace(day=1)

    curr_month_start = datetime.strftime(curr_month_start, '%Y-%m-%d')
    last_x_months_start = datetime.strftime(last_x_months_start, '%Y-%m-%d')

    return curr_month_start, last_x_months_start

def get_backtest_windows_date_list():
    """gets the total BACKTEST_MONTHS required for update and splits that into date ranges"""

    dates_list = []

    # ex '2022-03-01'    #'2022-02-01'   for filtering trades list
    # absolute_end_date, _ = helper_functions.get_last_x_from_start_of_month(months=BACKTEST_MONTHS)
    absolute_end_date, _ = get_last_x_from_start_of_month(months=BACKTEST_MONTHS)

    absolute_end_date = datetime.strptime(absolute_end_date, '%Y-%m-%d')

    month_end = absolute_end_date
    for i in range(0, BACKTEST_MONTHS):
        dates_dict = {}

        dates_dict['month_end'] = month_end
        month_start = (month_end - timedelta(days=1)).replace(day=1)
        dates_dict['month_start'] = month_start
        month_end = month_start

        dates_list.append(dates_dict)

    return dates_list

def date_filter_trades(trades,start_date,end_date):
    """
    filters trades where entry time > start date and end time < end date
    trades = [{},{}...]
    """
    df = pd.DataFrame(trades)
    df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])
    df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])
    df = df[(df['entry_datetime'] >= start_date) & (df['exit_datetime'] <= end_date)]
    df = df.reset_index(drop=True)
    df['trade_num'] = df.index + 1

    return df.to_dict('records')

def clean_price_data_DB_to_DF(df):
    keep_cols = ['datetime','open','high','low','close','volume']
    rename_col = {}
    for col in keep_cols:
        if col == 'datetime':
            rename_col[col] = 'Date_Time'
        else:
            rename_col[col] = col.capitalize()

    df = df[keep_cols]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.rename(columns=rename_col)
    df = df.sort_values(by='Date_Time', ascending=True)
    return df

def view_chart(data_df):
    """
    :param data_df: the input data as dataframe
    :param trade_data: the trade data as a list of dicts
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                 low=data_df['Low'], close=data_df['Close']), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=data_df['Date_Time'], y=data_df['auto_correl_1'], mode='lines', name='body highs',
                   line=dict(color='purple')),
        row=2, col=1)

    #fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

    # fig.add_trace(
    #     go.Scatter(x=data_df['Date_Time'], y=data_df['body_swing_high'], mode='markers', name='body highs',
    #                line=dict(color='purple')),
    #     row=1, col=1)
    #
    # fig.add_trace(
    #     go.Scatter(x=data_df['Date_Time'], y=data_df['body_swing_low'], mode='markers', name='body lows',
    #                line=dict(color='blue')),
    #     row=1, col=1)

    # fig.add_trace(
    #     go.Scatter(x=HL_df['Date_Time'], y=HL_df['body low'], mode='markers', name='body highs',
    #                line=dict(color='blue')),
    #     row=2, col=1)

    #     y0 = input_df['Volume'].max()
    #     y1 = input_df['Volume'].min()
    #
    #     fig.add_trace(
    #         go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9), name='entry'),
    #         row=2, col=1)
    #     # fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="green", width=2))
    #
    #     fig.update_layout(title=trade_data.ticker + ' - trade tags: ' + trade_data.trade_tags)

    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.show()

def get_file_metadata(filepath,type):
    type=type.lower()
    if type == 'created':
        ret_date = time.gmtime(os.path.getctime(filepath))
        ret_date = datetime.strptime(str(ret_date.tm_mday) + "-" + str(ret_date.tm_mon) + "-" + str(ret_date.tm_year), "%d-%m-%Y")
    if type == 'modified':
        ret_date = time.gmtime(os.path.getmtime(filepath))
        ret_date = datetime.strptime(str(ret_date.tm_mday) + "-" + str(ret_date.tm_mon) + "-" + str(ret_date.tm_year), "%d-%m-%Y")

    return ret_date

def needs_update(file_path, interval_in_days):

    if not exists(file_path):
        return True

    last_updated = get_file_metadata(filepath=file_path,type='created')
    curr_time = datetime.now()
    if (curr_time - last_updated).days > interval_in_days:
        return True

    return False

def candle_resampler(input_df, timeframe = str):
    """
    inputs origonal dataframe and selected timeframe and outputs dataframe of desired output conversion timeframe

    input dataframe input:

    Date, Open, High, Low, Close, Volume

    timeframe : '15min' , 1hr:'60min', 4hr:'240min', 1day:'1440min'

    """

    def fill_in(cols):
        volume = cols[0]
        target = cols[1]
        stock_close = cols[2]

        if volume == 0:
            target = stock_close

        return target

    #########################################

    if input_df['Date'].dtype != 'datetime64[ns]':
        # conver to datetime
        input_df['Date'] = pd.to_datetime(input_df['Date'])#, unit='ms'))

    input_df = input_df.set_index(pd.DatetimeIndex(input_df['Date']))

    data_ohlc = input_df.resample(timeframe).agg({'Open': 'first',
                                             'High': 'max',
                                             'Low': 'min',
                                             'Close': 'last',
                                             'Volume': 'sum'})

    data_ohlc = data_ohlc.reset_index()

    data_ohlc['Close'] = data_ohlc['Close'].fillna(method='ffill')
    data_ohlc['Open'] = data_ohlc[['Volume','Open','Close']].apply(fill_in,axis=1)
    data_ohlc['High'] = data_ohlc[['Volume', 'High', 'Close']].apply(fill_in, axis=1)
    data_ohlc['Low'] = data_ohlc[['Volume', 'Low', 'Close']].apply(fill_in, axis=1)



    #data_ohlc['Date_Time'] = data_ohlc['Date_Time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    return data_ohlc

def handicap_trades(trade_df, slippage=None, comission=None, carry=None):
    """
    penalize the outcome of trades by a selected amount to account for slippage/commission/carry
    input all handicaps as negative percentage of return for each individual trade

    !!!
    need to find a way to standardize the total investment, right now youre penalizing the dollar return which isnt realistic,
    penalize the total amount invested
    """

    trade_df['dollar_return'] = np.where(trade_df['bias'] == 'buy',
                                      (trade_df['exit_price'] - trade_df['entry_price']),
                                      (trade_df['entry_price'] - trade_df['exit_price']))

    # TODO need to figure out how to apply handicaps realistically
    # total_handicap = 0
    #
    # if slippage:
    #     slippage /= 100
    #     total_handicap += (slippage*dollar_return)
    #
    # if comission:
    #     comission /= 100
    #     total_handicap += (comission*dollar_return)
    #
    # if carry:
    #     carry /= 100
    #     total_handicap += (carry*dollar_return)
    #
    # trade_df['dollar_return'] = dollar_return + total_handicap

    return trade_df

def get_evaluation_metrics(class_obj,trade_data,candle_data):
    """
    trade data put in as list of dicts

    - strike rate
    - num trades
    - avg holding time
    - expectancy
    - avg winner and loser
    - largest winner and loser
    - equity curve data

    - max drawdown/drawup
    - kelly criterion
    - MAE/MFE

    put in comparative results to just holding long or short the entire time
    """

    import statsmodels.api as sm

    def get_MAE_MFE(frame):

        date_range_df = candle_data.loc[(candle_data['Date_Time'] > frame['entry_datetime']) & (candle_data['Date_Time'] <= frame['exit_datetime'])]

        if frame['bias'] == 'buy':

            curr_trade_MFE = (date_range_df['High'].max() - frame['entry_price']) / (
                    frame['entry_price'] - frame['risk_price'])
            curr_trade_MAE = (frame['entry_price'] - date_range_df['Low'].min()) / (
                    frame['entry_price'] - frame['risk_price'])

        elif frame['bias'] == 'sell':

            curr_trade_MAE = (date_range_df['High'].max() - frame['entry_price']) / (
                    frame['risk_price'] - frame['entry_price'])
            curr_trade_MFE = (frame['entry_price'] - date_range_df['Low'].min()) / (
                    frame['risk_price'] - frame['entry_price'])

        frame['MAE'] = curr_trade_MAE
        frame['MFE'] = curr_trade_MFE

        return frame

    def line_of_best_fit(frame):
        coef, b = np.polyfit(frame['trade_num'],frame['running_total_R'], 1)
        frame['line_of_best_fit'] = coef*frame['trade_num'] + b
        avg_error = abs(frame['line_of_best_fit'] - frame['running_total_R']).mean()
        # The standard error of the regression is the average distance that the observed values fall from the regression line.
        return coef,avg_error

    def get_returns_to_asset_corr(trades_df, candle_df):

        trades_df = trades_df[['exit_datetime','R_realized']]
        trades_df = trades_df.rename(columns={'exit_datetime': 'Date_Time'})

        merged_df = pd.merge_ordered(trades_df, candle_df, on='Date_Time')
        merged_df = merged_df.dropna(how='any')

        return merged_df['R_realized'].corr(merged_df['Close'])

    ########################################################################
    if not trade_data:
        return

    # put trade data into df
    trade_df = pd.DataFrame(trade_data)

    # convert times to datetimes
    trade_df['entry_datetime'] = pd.to_datetime(trade_df['entry_datetime'])
    trade_df['exit_datetime'] = pd.to_datetime(trade_df['exit_datetime'])

    trade_df = handicap_trades(trade_df=trade_df)

    # count number of trades
    num_trades = int(trade_df['trade_num'].count())

    trade_df['R_realized'] = np.where(trade_df['bias'] == 'buy',
                                      (trade_df['dollar_return'])/(trade_df['entry_price']-trade_df['risk_price']),
                                      (trade_df['dollar_return'])/(trade_df['risk_price']-trade_df['entry_price']))

    returns_to_asset_corr = get_returns_to_asset_corr(trades_df=trade_df, candle_df=candle_data)

    # get the total realized R for the trading session
    total_realized_R = trade_df['R_realized'].sum()

    trade_df['running_total_R'] = trade_df['R_realized'].cumsum()

    coef, avg_error = line_of_best_fit(frame=trade_df)

    trade_df['hold_time'] = (trade_df['exit_datetime'] - trade_df['entry_datetime']).dt.total_seconds()

    winners = trade_df['R_realized'][trade_df['R_realized'] > 0]
    losers = trade_df['R_realized'][trade_df['R_realized'] < 0]

    win_strike_rate = winners.count()/trade_df['trade_num'].count()
    loss_strike_rate = losers.count()/trade_df['trade_num'].count()

    avg_hold_time = trade_df['hold_time'].mean()

    avg_winner = winners.mean()
    avg_loser = losers.mean()

    largest_winner = winners.max()
    largest_loser = losers.min()

    trade_df['is_winner'] = np.where(trade_df['R_realized'] > 0,1,0)
    trade_df['is_loser'] = np.where(trade_df['R_realized'] < 0,1,0)

    consecutive_winner = trade_df['is_winner'].groupby((trade_df['is_winner'] != trade_df['is_winner'].shift()).cumsum()).transform('size') * trade_df['is_winner']
    consecutive_loser = trade_df['is_loser'].groupby((trade_df['is_loser'] != trade_df['is_loser'].shift()).cumsum()).transform('size') * trade_df['is_loser']

    max_drawup = int(consecutive_winner.max())
    max_drawdown = int(consecutive_loser.max())

    expectancy = (avg_winner * win_strike_rate) - abs(avg_loser * loss_strike_rate)

    # MFE/MAE
    trade_df['MFE'] = None
    trade_df['MAE'] = None
    trade_df = trade_df.apply(get_MAE_MFE,axis=1)

    winners_avg_MAE = (trade_df[trade_df['R_realized'] > 0]['MAE'].mean()) * -1
    winners_avg_MFE = trade_df[trade_df['R_realized'] > 0]['MFE'].mean()

    winners_std_dev_MAE = trade_df[trade_df['R_realized'] > 0]['MAE'].std()
    winners_std_dev_MFE = trade_df[trade_df['R_realized'] > 0]['MFE'].std()

    losers_avg_MAE = (trade_df[trade_df['R_realized'] < 0]['MAE'].mean()) * -1
    losers_avg_MFE = trade_df[trade_df['R_realized'] < 0]['MFE'].mean()

    losers_std_dev_MAE = trade_df[trade_df['R_realized'] < 0]['MAE'].std()
    losers_std_dev_MFE = trade_df[trade_df['R_realized'] < 0]['MFE'].std()

    # handles divide by zero error
    if winners.sum() == 0 or losers.sum() == 0:
        kelly_criterion = None
    else:
        kelly_criterion = win_strike_rate - ((1-win_strike_rate)/(winners.sum()/abs(losers.sum())))

    class_obj.number_of_bars = len(candle_data.loc[(candle_data['Date_Time'] >= class_obj.period_start_date) &
                                                   (candle_data['Date_Time'] <= class_obj.period_end_date)])
    class_obj.total_realized_R = total_realized_R
    class_obj.strike_rate = win_strike_rate
    class_obj.num_trades = num_trades
    class_obj.avg_hold_time = avg_hold_time
    class_obj.expectancy = expectancy
    class_obj.avg_winner = avg_winner
    class_obj.avg_loser = avg_loser
    class_obj.largest_winner = largest_winner
    class_obj.largest_loser = largest_loser
    class_obj.max_drawdown = max_drawdown
    class_obj.max_drawup = max_drawup
    class_obj.winners_avg_MAE = winners_avg_MAE
    class_obj.winners_avg_MFE = winners_avg_MFE
    class_obj.winners_std_dev_MAE = winners_std_dev_MAE
    class_obj.winners_std_dev_MFE = winners_std_dev_MFE
    class_obj.losers_avg_MAE = losers_avg_MAE
    class_obj.losers_avg_MFE = losers_avg_MFE
    class_obj.losers_std_dev_MAE = losers_std_dev_MAE
    class_obj.losers_std_dev_MFE = losers_std_dev_MFE
    class_obj.kelly_criterion = kelly_criterion
    class_obj.returns_to_asset_corr = returns_to_asset_corr
    class_obj.equity_curve_regression_slope = coef
    class_obj.equity_curve_regression_std_error = avg_error

    drop_cols = ['dollar_return', 'line_of_best_fit','hold_time','is_winner','is_loser'] #,'MFE','MAE']
    trade_df = trade_df.drop(drop_cols, axis=1)
    trade_df = trade_df.round({'risk_price':5,'R_realized':2,'running_total_R':2,'MAE':2,'MFE':2})

    class_obj.trades_list = trade_df.to_dict(orient='records')

    return class_obj

def compare_two_lists(regular, abridged):
    """
    Compare two lists and logs the difference.
    :param regular: first list.
    :param abridged: second list.
    :return:      if there is difference between both lists.
    """
    regular.sort(key=lambda item:item['entry_datetime'])
    abridged.sort(key=lambda item:item['entry_datetime'])

    for i in regular:
        if not i in abridged:
            print(i)
            for j in abridged:
                if i['trade_num'] == j['trade_num']:
                    print('regular   ',i)
                    print('abridged  ',j)

                    print('-----------------------------------------------------------------------------------------')
                    print('-----------------------------------------------------------------------------------------')
                    print('-----------------------------------------------------------------------------------------')

def order_params_dict(strat_params):
    """
    orders dictionary by keys then converts to string to put into DB
    is ordered for searchability in case the order to the strat params dict gets changed
    """

    new_list = [[k, v] for k, v in strat_params.items()]
    new_list.sort(key=lambda x: x[0])

    return dict(new_list)

def create_json_string(input):
    def converter(obj):
        return obj.__str__()

    ######################################
    return json.dumps(input, default=converter)

def get_core_num(pct,physical=True):
    import psutil

    logical = True
    if physical:
        logical=False

    cores = int((psutil.cpu_count(logical=logical) * pct))
    if cores < 1:
        cores = 1
    return cores

