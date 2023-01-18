import itertools
import statistics
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import helper_functions
import technical_indicators
import time

STATIC_TRADE_METRICS = ['trade_num', 'bias', 'risk_price', 'entry_datetime', 'entry_price', 'exit_datetime', 'exit_price']
ATR_NUM_RANGE = list(range(15, 55, 5)) #
ATR_MULT_RANGE = [x / 10.0 for x in range(6, 32, 4)] #
STD_DEV_RANGE = list(range(14, 26, 3)) + list(range(28, 54, 5)) #-3
STD_DEV_MULT_RANGE = [x / 10.0 for x in range(6, 32, 4)] #-6
BINARY_RANGE =  list(range(0, 2)) #[1]
SINGLE_MA_RANGE = list(range(10, 22, 3)) + list(range(22, 55, 5)) + list(range(60, 100, 10)) + list(range(100, 200, 20))
SLOW_MA = list(range(50, 75, 5)) + list(range(80, 120, 10)) #
FAST_MA = list(range(6, 19, 3)) + list(range(23, 44, 5)) # -3
R_PROF_TARG_RANGE = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0] #
BIAS_MA = [40] + list(range(55,100,15)) + list(range(120,180,20)) + [200] #-3
RSI = list(range(4, 28, 3))
RSI_UPPER_BIAS = [60, 70, 80, 90] # [[40,60], [30,70], [20,80], [10,90]]
RSI_LOWER_BIAS = [40,30,20,10] #list(range(10, 48, 4))
RVI = list(range(2, 40, 4))
RVI_SIGNAL = list(range(2, 15, 3)) #TODO, no fucking idea what this does
RVI_STD_DEV_NUM = list(range(14, 26, 3)) + list(range(28, 54, 5))
MACD_SIGNAL = list(range(3, 13, 3)) + list(range(15, 31, 5))
HEIKEN_ASHI_RANGE = list(range(2,16))

class trade_metrics():
    trade_num = None
    bias = None
    risk_price = None  # risk in terms of local price amounts
    entry_datetime = None
    entry_price = None
    exit_datetime = None
    exit_price = None

def create_permuations(params_dict):
    """create permutations"""
    def sort_params_list(perms_list):

        exec_str = 'ret_list = sorted(perms_list, key=lambda t: ('

        for enum, i in enumerate(params_dict):
            pos = (len(params_dict) - enum) - 1
            exec_str += 't[' + str(pos) + '],'

        exec_str = exec_str.rstrip(exec_str[-1]) + '))'

        exec(exec_str)

        return locals()['ret_list']

    ################################################################
    # now = time.time()
    param_perm_list = list(params_dict.values())

    input_params_perm = []
    for k in list(itertools.product(*param_perm_list)):
        input_params_perm.append(k)

    return input_params_perm

def create_permuations_labels(params_dict):
    return list(params_dict.keys())

def create_params_dict(strat_params, perm):
    params_name_list = list(strat_params.keys())
    ret_params = {params_name_list[i]: perm[i] for i in range(len(params_name_list))}

    return ret_params

def adjust_for_slow_fast_MAs(input_perms, slow_idx, fast_idx):
    # for perms that rely on a slow and fast MA
    ret_perms = []
    for perm in input_perms:
        if perm[slow_idx] <= perm[fast_idx]:
            continue
        ret_perms.append(perm)
    return ret_perms

def adjust_for_RSI_bias_ranges(input_perms,upper_idx,lower_idx):

    bias_combos = [(RSI_UPPER_BIAS[i], RSI_LOWER_BIAS[i]) for i in range(0, len(RSI_UPPER_BIAS))]

    ret_perms = []
    for perm in input_perms:
        perm_bias_combos = (perm[upper_idx],perm[lower_idx])
        if perm_bias_combos in bias_combos:
            ret_perms.append(perm)

    return ret_perms

########################################################################################################################
########################################################################################################################
########################################################################################################################

def basic_X_MA_crossover_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    simple 1 moving average crossover strategy

    enter long on candle close when price crosses above, switch on cross below

    exit on candle close after either ATR multiple loss is reached or crossover occurs

    base risk/judge reward off X atr multiple

    trades will not overlap, 1 trade at a time
    """

    # strat_params = {'MA': [10, 13, 16, 19, 22, 27, 32, 42, 52, 60, 80, 100, 120, 180],
    #                 'ATR_num': [15, 20, 25, 30, 35, 40, 45, 50],
    #                 #'ATR_mult': [0.6, 1.0, 1.5, 2.0, 3.0],
    #                 'use_SMA': BINARY_RANGE}

    strat_params = {'MA': [x for x in range(10, 200, 4)],
                    'ATR_num': [x for x in range(10, 40)],
                    # 'ATR_mult': [0.6, 1.0, 1.5, 2.0, 3.0],
                    'use_SMA': BINARY_RANGE}



    ## custom
    # strat_params = {'MA': [11],
    #                 'ATR_num': [15],
    #                 'ATR_mult': [14],
    #                 'use_SMA':[1]}

    # test
    # strat_params = {'MA': list(range(14, 15)),
    #           'ATR_num': list(range(14,15)),
    #           'ATR_mult': [14],
    #           'use_SMA': BINARY_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k:v[0] for (k,v) in strat_params.items()}

        ret_params = {}
        for k,v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':

        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    # else:
    #     params = create_params_dict(strat_params=strat_params, perm=params)

    ##################################################################################

    class df_col_headers:
        MA_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""

        # labels SMA/EMA
        if params['use_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['MA'])
            df_col_headers.MA_header = str(params['MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['MA'])
            df_col_headers.MA_header = str(params['MA']) + 'EMA'

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.MA_header], mode='lines', name=df_col_headers.MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):

            curr_trade = trade_metrics()

            # create current trade object
            curr_trade.trade_num = len(trades_list.completed)+1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                return 'buy'

            if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                return 'sell'

            return False

        def exit_criteria_met(frame,trade_obj):
            if frame['curr_bias'] == 'buy':
                if frame['Close'] < trade_obj.risk_price or frame['prev_bias'] == 'sell':
                    return True

            if frame['curr_bias'] == 'sell':
                if frame['Close'] > trade_obj.risk_price or frame['prev_bias'] == 'buy':
                    return True

            return False

        ###################################################################
        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame,trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame,trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

    ###########################################################
    data_df = do_indicator_calculations(data_df=data_df)

    # get the bias based on crossover
    data_df['curr_bias'] = np.where(data_df['Close'] > data_df[df_col_headers.MA_header], 'buy','sell')
    data_df['prev_bias'] = data_df['curr_bias'].shift(1)

    data_df.apply(get_trades, axis=1)

    ######################################################################

    if do_chart:
        view_chart(trade_data=completed_trades_list)

    return trades_list.completed

def basic_X_MA_crossover_V2_ABRIDGED(params,data_df,do_chart=False):
    """
    simple 1 moving average crossover strategy

    enter long on candle close when price crosses above
    emter short on candle close when price crosses below

    loss exit on ATR multiple
    profit exit on a R levels

    base risk/judge reward off X atr multiple

    trades will overlap (limit or 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'MA': [10, 16, 22, 28, 36, 46, 56, 70, 90, 120, 180],
                    'use_SMA': BINARY_RANGE,
                    'ATR_num': [15, 20, 30, 50],
                    #'ATR_mult': [1.0, 1.5, 2.0, 3.0],
                    'R_mult_PT': [1.0, 1.5, 2.0, 3.0, 5.0]}

    ################################################################################
    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        MA_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""

        # SMA/EMA
        if params['use_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['MA'])
            df_col_headers.MA_header = str(params['MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['MA'])
            df_col_headers.MA_header = str(params['MA']) + 'EMA'

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.MA_header], mode='lines', name=df_col_headers.MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k,v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                return 'buy'

            if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                return 'sell'

            return False

        def exit_criteria_met(frame,trade_obj):
            # has the profit target or loss exit been hit
            if trade_obj.bias == 'buy':
                if frame['Close'] < trade_obj.risk_price or frame['Close'] > trade_obj.profit_target:
                    return True

            elif trade_obj.bias == 'sell':
                if frame['Close'] > trade_obj.risk_price or frame['Close'] < trade_obj.profit_target:
                    return True

            return False

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    # get the bias based on crossover
    data_df['curr_bias'] = np.where(data_df['Close'] > data_df[df_col_headers.MA_header], 'buy', 'sell')
    data_df['prev_bias'] = data_df['curr_bias'].shift(1)

    data_df.apply(get_trades, axis=1)

    #########################################################

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_XY_MA_crossover_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    simple 2 moving average crossover strategy

    enter long on candle close when fast crosses above slow
    enter short on candle close when fast crosses below slow

    exit on candle close after either ATR multiple loss is reached or crossover occurs

    base risk/judge reward off X atr multiple

    trades will not overlap, 1 trade at a time

    """

    strat_params = {'slow_MA':[30, 40, 60, 70, 90],
                    'fast_MA': [9, 18, 28, 34, 43],
                    'use_slow_SMA':BINARY_RANGE,
                    'use_fast_SMA':BINARY_RANGE,
                    #'ATR_mult': [0.6, 1.0, 1.5, 2.0, 3.0],
                    'ATR_num':[15, 20, 30, 40, 50]}

    ################################################################################
    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)

        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def check_MAs(params):
        if params['fast_MA'] >= params['MA']:
            print('FIX YOUR MOVING AVERAGE NUMBERS')
            exit()

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""

        # SMA/EMA
        if params['use_slow_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        if params['use_fast_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'EMA'

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines', name=df_col_headers.slow_MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines', name=df_col_headers.fast_MA_header,
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):
        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                return 'buy'

            if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):
            # has a crossover occured?
            if frame['prev_bias'] and frame['curr_bias'] != frame['prev_bias']:
                return True

            #otherwise has the risk profile been broken?
            elif trade_obj.bias == 'buy' and frame['Close'] <= trade_obj.risk_price:
                return True
            elif trade_obj.bias == 'sell' and frame['Close'] >= trade_obj.risk_price:
                return True

            else:
                return False

        ###################################################################

        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame,trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame,trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

    ###########################################################
    # check_MAs(params=params)

    data_df = do_indicator_calculations(data_df=data_df)

    # get the bias based on crossover
    data_df['curr_bias'] = np.where(data_df[df_col_headers.fast_MA_header] > data_df[df_col_headers.slow_MA_header], 'buy','sell')
    data_df['prev_bias'] = data_df['curr_bias'].shift(1)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_XY_MA_crossover_V2_ABRIDGED(params,data_df,do_chart=False):
    """
    simple 2 moving average crossover strategy

    enter long on candle close when fast crosses above slow
    enter short on candle close when fast crosses below slow

    loss exit on ATR multiple
    profit exit on a R levels

    base risk/judge reward off X atr multiple

    trades will overlap (limit or 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'slow_MA':SLOW_MA,
                    'fast_MA': FAST_MA,
                    'use_slow_SMA':BINARY_RANGE,
                    'use_fast_SMA':BINARY_RANGE,
                    'ATR_num':list(range(15, 30, 5))+list(range(30, 50, 10)),
                    #'ATR_mult':[x / 10.0 for x in range(6, 11, 4)] + [x / 10.0 for x in range(15, 21, 5)] + [3.0],
                    'R_mult_PT':[1.0, 2.0, 3.0, 5.0, 7.0, 9.0]}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)

        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def check_MAs(params):
        if params['fast_MA'] >= params['MA']:
            print('FIX YOUR MOVING AVERAGE NUMBERS')
            exit()

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""

        # SMA/EMA
        if params['use_slow_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        if params['use_fast_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'EMA'

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines', name=df_col_headers.slow_MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines', name=df_col_headers.fast_MA_header,
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):
        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                return 'buy'

            if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                return 'sell'

            return False

        def exit_criteria_met(frame,trade_obj):
            # has the profit target or loss exit been hit
            if trade_obj.bias == 'buy':
                if frame['Close'] < trade_obj.risk_price or frame['Close'] > trade_obj.profit_target:
                    return True

            elif trade_obj.bias == 'sell':
                if frame['Close'] > trade_obj.risk_price or frame['Close'] < trade_obj.profit_target:
                    return True

            return False

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    # check_MAs(params=params)

    data_df = do_indicator_calculations(data_df=data_df)

    # get the bias based on crossover
    data_df['curr_bias'] = np.where(data_df[df_col_headers.fast_MA_header] > data_df[df_col_headers.slow_MA_header], 'buy', 'sell')
    data_df['prev_bias'] = data_df['curr_bias'].shift(1)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def XYZ_MA_crossover_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    3 moving average crossover strategy
    bias MA provides long/short bias (only take longs during buy bias and vice versa)


    enter trade direcitonal bias on MA crossovers
    exit on candle close after either ATR multiple loss is reached or opposite crossover occurs

    base risk/judge reward off X atr multiple

    trades will not overlap, 1 trade at a time

    params = indicator_params()
    params.slow_MA = 20
    params.use_slow_SMA = 1
    params.fast_MA = 9
    params.use_fast_SMA = 0
    params.bias_MA = 200
    params.use_bias_SMA = 0
    params.ATR_num = 14

    ###################################################
    """

    strat_params = {'slow_MA':[50, 60, 70, 80, 90],
                    'fast_MA': [12, 15, 28, 35, 43],
                    'use_slow_SMA':BINARY_RANGE,
                    'use_fast_SMA':BINARY_RANGE,
                    'bias_MA':[30, 50, 90, 140, 200],
                    'use_bias_SMA':BINARY_RANGE,
                    # 'ATR_mult': [1.0, 1.5, 2.0, 3.0],
                    'ATR_num': [15, 30, 50]}
    ################################################################################
    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)

        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        bias_MA_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def check_MAs(params):
        if params['fast_MA'] >= params['MA']:
            print('FIX YOUR MOVING AVERAGE NUMBERS')
            exit()

    def do_indicator_calculations(data_df):
        """
        calculate indicators
        get the indicator headers
        chop data down to the slowest one so the strategy starts on all updated data
        """

        # slow MA
        if params['use_slow_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        # fast MA
        if params['use_fast_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'EMA'

        # bias MA
        if params['use_bias_SMA'] == 1: # use SMA
            data_df = technical_indicators.simple_moving_average(df=data_df,MA_num=params['bias_MA'])
            df_col_headers.bias_MA_header = str(params['bias_MA']) + 'SMA'
        else: # use EMA
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['bias_MA'])
            df_col_headers.bias_MA_header = str(params['bias_MA']) + 'EMA'

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def do_bias_calculations(df):
        # get the long/short bias based on which side of the bias MA price is on
        df['long_short'] = np.where(df[df_col_headers.bias_MA_header] < df['Close'], 'long', 'short')

        # get the bias based on crossover
        df['curr_bias'] = np.where(df[df_col_headers.fast_MA_header] > df[df_col_headers.slow_MA_header],'buy', 'sell')
        df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['long_short'] == 'long':
                if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame['long_short'] == 'short':
                if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):
            # has a crossover occured?
            if frame['prev_bias'] and frame['curr_bias'] != frame['prev_bias']:
                return True

            #otherwise has the risk profile been broken?
            elif trade_obj.bias == 'buy' and frame['Close'] <= trade_obj.risk_price:
                return True
            elif trade_obj.bias == 'sell' and frame['Close'] >= trade_obj.risk_price:
                return True

            else:
                return False

        ###################################################################

        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame,trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame,trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines', name=df_col_headers.slow_MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines', name=df_col_headers.fast_MA_header,
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.bias_MA_header], mode='lines',
                       name=df_col_headers.fast_MA_header,
                       line=dict(color='black')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    ######################################################################
    data_df = do_indicator_calculations(data_df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def XYZ_MA_crossover_V2_ABRIDGED(params, data_df, do_chart=False):
    """
    3 moving average crossover strategy
    bias MA provides long/short bias (only take longs during long bias and vice versa)

    enter trade direcitonal bias on MA crossovers

    loss exit on ATR multiple
    profit exit on a R levels

    base risk/judge reward off X atr multiple

    trades will overlap (limit or 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times

    """

    strat_params = {'slow_MA': list(range(50, 85, 5)),
                    'fast_MA': FAST_MA,
                    'use_slow_SMA': BINARY_RANGE,
                    'use_fast_SMA': BINARY_RANGE,
                    'bias_MA': BIAS_MA,
                    'use_bias_SMA': BINARY_RANGE,
                    'ATR_num': list(range(15, 30, 5)) + list(range(30, 50, 10)),
                    #'ATR_mult': [x / 10.0 for x in range(6, 11, 4)] + [x / 10.0 for x in range(15, 21, 5)] + [3.0],
                    'R_mult_PT': [1.0, 2.0, 4.0, 6.5, 9.0]}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)

        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        bias_MA_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def check_MAs(params):
        if params['fast_MA'] >= params['MA']:
            print('FIX YOUR MOVING AVERAGE NUMBERS')
            exit()

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # SMA/EMA
        # slow MA
        if params['use_slow_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        # fast MA
        if params['use_fast_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'EMA'

        # bias MA
        if params['use_bias_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['bias_MA'])
            df_col_headers.bias_MA_header = str(params['bias_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['bias_MA'])
            df_col_headers.bias_MA_header = str(params['bias_MA']) + 'EMA'

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):
        # get the long/short bias based on which side of the bias MA price is on
        df['long_short'] = np.where(df[df_col_headers.bias_MA_header] < df['Close'], 'long', 'short')

        # get the bias based on crossover
        df['curr_bias'] = np.where(df[df_col_headers.fast_MA_header] > df[df_col_headers.slow_MA_header], 'buy', 'sell')
        df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines',
                       name=df_col_headers.slow_MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines',
                       name=df_col_headers.fast_MA_header,
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['long_short'] == 'long':
                if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame['long_short'] == 'short':
                if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame,trade_obj):
            # has the profit target or loss exit been hit
            if trade_obj.bias == 'buy':
                if frame['Close'] < trade_obj.risk_price or frame['Close'] > trade_obj.profit_target:
                    return True

            elif trade_obj.bias == 'sell':
                if frame['Close'] > trade_obj.risk_price or frame['Close'] < trade_obj.profit_target:
                    return True

            return False

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################

    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def RSI_moving_avg_crossover_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    Get bias from RSI then use moving averages for entries
    base risk/judge reward off X atr multiple

    enter on crossovers
    exit on crossovers or if ATR risk level is hit

    trades will not overlap, 1 trade at a time
    """

    strat_params = {'RSI_upper_bias':RSI_UPPER_BIAS,
                    'RSI_lower_bias':RSI_LOWER_BIAS,
                    'slow_MA':list(range(50, 85, 5)),
                    'fast_MA': FAST_MA,
                    'use_slow_SMA':BINARY_RANGE,
                    'use_fast_SMA':BINARY_RANGE,
                    'RSI_num':RSI,
                    #'ATR_mult': [x / 10.0 for x in range(6, 11, 4)] + [x / 10.0 for x in range(15, 21, 5)] + [3.0]
                    'ATR_num': list(range(15, 30, 5)) + list(range(30, 50, 10))}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)
        input_params_perm = adjust_for_RSI_bias_ranges(input_perms=input_params_perm,
                                                       upper_idx=0,
                                                       lower_idx=1)

        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        RSI_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(df):
        """
        calculate indicators
        get the indicator headers
        chop data down to the slowest one so the strategy starts on all updated data
        """

        # slow MA
        if params['use_slow_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        # fast MA
        if params['use_fast_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'EMA'

        # RSI
        df = technical_indicators.RSI(df=df, periods=params['RSI_num'])
        df_col_headers.RSI_header = str(params['RSI_num']) + 'RSI'

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):
        # get the long/short bias based on which side of the RSI price is on
        conditions = [df[df_col_headers.RSI_header] >= params['RSI_upper_bias'][1],
                      df[df_col_headers.RSI_header] < params['RSI_lower_bias'][0]]
        choices = ["short", 'long']
        df["long_short"] = np.select(conditions, choices, default='neutral')

        # get the bias based on crossover
        df['curr_bias'] = np.where(df[df_col_headers.fast_MA_header] > df[df_col_headers.slow_MA_header],'buy', 'sell')
        df['prev_bias'] = df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines', name=df_col_headers.slow_MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines', name=df_col_headers.fast_MA_header,
                       line=dict(color='purple')),
            row=1, col=1)

        # add volume
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=True)
        # add RSI
        fig.add_trace(go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.RSI_header]), row=2, col=1, secondary_y=False)
        # add upper and lower bounds of RSI
        data_df['RSI upper'] = params['RSI_upper_bias']
        data_df['RSI lower'] = params['RSI_lower_bias']
        fig.add_trace(go.Scatter(x=data_df['Date_Time'], y=data_df['RSI upper'], mode='lines',line=dict(color='red')), row=2, col=1,secondary_y=False)

        fig.add_trace(go.Scatter(x=data_df['Date_Time'], y=data_df['RSI lower'], mode='lines',
                       line=dict(color='green')), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):
        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['long_short'] == 'long' and frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame['long_short'] == 'short' and frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):
            # has a crossover occured?
            if frame['prev_bias'] and frame['curr_bias'] != frame['prev_bias']:
                return True

            #otherwise has the risk profile been broken?
            elif trade_obj.bias == 'buy' and frame['Close'] <= trade_obj.risk_price:
                return True

            elif trade_obj.bias == 'sell' and frame['Close'] >= trade_obj.risk_price:
                return True

            else:
                return False

        ###################################################################

        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame,trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame,trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

    ###############################################################################################
    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def RSI_moving_avg_crossover_V2_ABRIDGED(params, data_df, do_chart=False):
    """
    Get bias from RSI then use moving averages for entries
    base risk/judge reward off X atr multiple

    enter on candle close on crossovers

    loss exit on ATR multiple
    profit exit on a R levels

    base risk/judge reward off X atr multiple

    trades will overlap (limit or 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times

    """

    strat_params = {'slow_MA':SLOW_MA,
                    'fast_MA': FAST_MA,
                    'use_slow_SMA':BINARY_RANGE,
                    'use_fast_SMA':BINARY_RANGE,
                    'RSI_num':RSI,
                    'RSI_upper_bias':RSI_UPPER_BIAS,
                    'RSI_lower_bias':RSI_LOWER_BIAS,
                    'ATR_num':ATR_NUM_RANGE,
                    #'ATR_mult':ATR_MULT_RANGE,
                    'R_mult_PT':R_PROF_TARG_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)

        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        RSI_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def check_MAs(params):
        if params['fast_MA'] >= params['MA']:
            print('FIX YOUR MOVING AVERAGE NUMBERS')
            exit()

    def do_indicator_calculations(df):
        """
        calculate indicators
        get the indicator headers
        chop data down to the slowest one so the strategy starts on all updated data
        """

        # slow MA
        if params['use_slow_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        # fast MA
        if params['use_fast_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'EMA'

        # RSI
        df = technical_indicators.RSI(df=df, periods=params['RSI_num'])
        df_col_headers.RSI_header = str(params['RSI_num']) + 'RSI'

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):
        # get the long/short bias based on which side of the RSI price is on
        conditions = [df[df_col_headers.RSI_header] >= params['RSI_upper_bias'],
                      df[df_col_headers.RSI_header] < params['RSI_lower_bias']]
        choices = ['short', 'long']
        df["long_short"] = np.select(conditions, choices, default='neutral')

        # get the bias based on crossover
        df['curr_bias'] = np.where(df[df_col_headers.fast_MA_header] > df[df_col_headers.slow_MA_header],'buy', 'sell')
        df['prev_bias'] = df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines', name=df_col_headers.slow_MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines', name=df_col_headers.fast_MA_header,
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['long_short'] == 'long' and frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame['long_short'] == 'short' and frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):
            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    PT = trade_obj.entry_price + (
                                abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                    if frame['Close'] > PT:
                        return True

                if trade_obj.bias == 'sell':
                    PT = trade_obj.entry_price - (
                                abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                    if frame['Close'] < PT:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################

    #check_MAs(params=params)

    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades,axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def RVI_crossover_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    buy on crossover of RVI above signal line, sell on opposite

    enter on crossovers
    exit on crossovers or if ATR risk level is hit

    trades will not overlap, 1 trade at a time
    """

    strat_params = {'RVI_num':[6, 10, 18, 25, 34, 48],
                    'RVI_signal':[5, 8, 11, 14],
                    # 'ATR_mult':[0.6, 1.0, 1.5, 2.0, 3.0],
                    'ATR_num': [15, 20, 30, 40, 50]}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        RVI_header = None
        RVI_signal_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""
        # RVI
        data_df = technical_indicators.RVI(df=data_df,periods=params['RVI_num'],signal_MA=params['RVI_signal'])
        df_col_headers.RVI_header = str(params['RVI_num']) + 'RVI'
        df_col_headers.RVI_signal_header = str(params['RVI_num']) + 'RVI_signal'

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1,specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.RVI_header], mode='lines', name=df_col_headers.RVI_header,
                       line=dict(color='orange')),
            row=2, col=1,secondary_y=True)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.RVI_signal_header], mode='lines', name=df_col_headers.RVI_signal_header,
                       line=dict(color='purple')),
            row=2, col=1,secondary_y=True)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):
        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):
            # has a crossover occured?
            if frame['prev_bias'] and frame['curr_bias'] != frame['prev_bias']:
                return True

            #otherwise has the risk profile been broken?
            elif trade_obj.bias == 'buy' and frame['Close'] <= trade_obj.risk_price:
                return True

            elif trade_obj.bias == 'sell' and frame['Close'] >= trade_obj.risk_price:
                return True

            else:
                return False

        ###################################################################

        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame,trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame,trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    data_df['curr_bias'] = np.where(data_df[df_col_headers.RVI_header] > data_df[df_col_headers.RVI_signal_header],'buy','sell')
    data_df['prev_bias'] = data_df['curr_bias'].shift(1)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def RVI_crossover_V2_ABRIDGED(params, data_df, do_chart=False):
    """
    buy on crossover of RVI above signal line, sell on opposite

    loss exit on ATR multiple
    profit exit on a R levels

    base risk/judge reward off X atr multiple

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times

    """

    strat_params = {'RVI_num':[6, 10, 16, 24, 34, 48],
                    'RVI_signal':[5, 8, 11, 14],
                    'ATR_num':[15, 30, 50],
                    #'ATR_mult':[1.0, 1.5, 2.0, 3.0],
                    'R_mult_PT':[1.0, 1.5, 2.0, 3.0, 5.0]}

    ################################################################################
    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        RVI_header = None
        RVI_signal_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""
        # RVI
        data_df = technical_indicators.RVI(df=data_df,periods=params['RVI_num'],signal_MA=params['RVI_signal'])
        df_col_headers.RVI_header = str(params['RVI_num']) + 'RVI'
        df_col_headers.RVI_signal_header = str(params['RVI_num']) + 'RVI_signal'

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1,specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.RVI_header], mode='lines', name=df_col_headers.RVI_header,
                       line=dict(color='orange')),
            row=2, col=1,secondary_y=True)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.RVI_signal_header], mode='lines', name=df_col_headers.RVI_signal_header,
                       line=dict(color='purple')),
            row=2, col=1,secondary_y=True)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):
            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] > trade_obj.profit_target:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] < trade_obj.profit_target:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    data_df = do_indicator_calculations(data_df=data_df)

    # get the bias based on crossover
    data_df['curr_bias'] = np.where(data_df[df_col_headers.RVI_header] > data_df[df_col_headers.RVI_signal_header], 'buy', 'sell')
    data_df['prev_bias'] = data_df['curr_bias'].shift(1)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def RVI_crossover_V3_ABRIDGED(params, data_df, do_chart=False):
    """
    get rolling period std deviation for RVI
    get long/short bias based on if RVI is outside of X std deviations
        - buy on crossover of RVI above signal if long bias
        - sell on crossover of RVI below signal if short bias

    loss exit on ATR multiple
    profit exit on a R levels

    base risk/judge reward off X atr multiple

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """


    strat_params = {'RVI_num':RVI,
                  'RVI_signal':RVI_SIGNAL,
                  'RVI_std_dev_num':RVI_STD_DEV_NUM, # for rolling std dev for RVI
                  'RVI_std_dev_cutoff':STD_DEV_MULT_RANGE, # if RVI is +/- X std devs assign long/short bias
                  'ATR_num': list(range(15, 30, 5)) + list(range(30, 50, 10)),
                  #'ATR_mult': [x / 10.0 for x in range(6, 11, 4)] + [x / 10.0 for x in range(15, 21, 5)] + [3.0],
                  'R_mult_PT':[1.0, 1.5, 2.0, 3.0, 5.0, 7.0]}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        RVI_header = None
        RVI_signal_header = None
        RVI_std_dev_header = None
        RVI_mean_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""
        # RVI
        df = technical_indicators.RVI(df=df, periods=params['RVI_num'], signal_MA=params['RVI_signal'])
        df_col_headers.RVI_header = str(params['RVI_num']) + 'RVI'
        df_col_headers.RVI_signal_header = str(params['RVI_num']) + 'RVI_signal'

        # std dev of RVI
        df_col_headers.RVI_std_dev_header = str(params['RVI_std_dev_num']) + 'RVI std dev'
        df[df_col_headers.RVI_std_dev_header] = df[df_col_headers.RVI_header].rolling(params['RVI_std_dev_num']).std()

        # rolling avg mean of RVI
        df_col_headers.RVI_mean_header = str(params['RVI_std_dev_num']) + 'RVI mean'
        df[df_col_headers.RVI_mean_header] = df[df_col_headers.RVI_header].rolling(params['RVI_std_dev_num']).mean()


        # upper/lower bounds for RVI std dev
        df['RVI upper bound'] = df[str(params['RVI_std_dev_num'])+'RVI mean'] + (df[df_col_headers.RVI_std_dev_header]*params['RVI_std_dev_cutoff'])
        df['RVI lower bound'] = df[str(params['RVI_std_dev_num'])+'RVI mean'] - (df[df_col_headers.RVI_std_dev_header]*params['RVI_std_dev_cutoff'])

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):

        # get the long/short bias based on which side of the RSI price is on
        conditions = [df[df_col_headers.RVI_header] >= df['RVI upper bound'],
                      df[df_col_headers.RVI_header] < df['RVI lower bound']]
        choices = ["short", 'long']
        df["long_short"] = np.select(conditions, choices, default='neutral')

        # get the bias based on crossover
        df['curr_bias'] = np.where(df[df_col_headers.RVI_header] > df[df_col_headers.RVI_signal_header],'buy', 'sell')
        df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1,specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # RVI
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.RVI_header], mode='lines', name=df_col_headers.RVI_header,
                       line=dict(color='orange')),
            row=2, col=1,secondary_y=True)

        # RVI signal
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.RVI_signal_header], mode='lines', name=df_col_headers.RVI_signal_header,
                       line=dict(color='purple')),
            row=2, col=1,secondary_y=True)

        # RVI mean
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.RVI_mean_header], mode='lines',
                       name=df_col_headers.RVI_mean_header,line=dict(color='purple')),
            row=2, col=1, secondary_y=True)

        # RVI std dev upper and lower bounds
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['RVI upper bound'], mode='lines',
                       name='RVI upper bound', line=dict(color='purple')),
            row=2, col=1, secondary_y=True)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['RVI lower bound'], mode='lines',
                       name='RVI lower bound', line=dict(color='purple')),
            row=2, col=1, secondary_y=True)



        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame["long_short"] == 'long' and frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame["long_short"] == 'short' and frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):

            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] > trade_obj.profit_target:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] < trade_obj.profit_target:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_MACD_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    if MACD crosses above the signal line then buy, else sell/reverse

    profit exit on crossover
    loss exit on crossover or if ATR risk is hit

    trades will not overlap
    """

    strat_params = {'slow_MA':list(range(26, 75, 5)),
                  'fast_MA':FAST_MA,
                  'signal':MACD_SIGNAL,
                  #'ATR_mult': [x / 10.0 for x in range(6, 11, 4)] + [x / 10.0 for x in range(15, 21, 5)] + [3.0]
                  'ATR_num':list(range(15, 31, 5)) + [40,50]}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        signal_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def check_MAs(params):
        if params['fast_MA'] >= params['slow_MA']:
            print('FIX YOUR MOVING AVERAGE NUMBERS')
            exit()

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # MACD
        df = technical_indicators.MACD(df=df, slow_MA=params['slow_MA'], fast_MA=params['fast_MA'], signal=params['signal'])
        df_col_headers.signal_header = str(params['signal'])+'signal_line'
        df_col_headers.fast_MA_header = str(params['fast_MA'])+'EMA'
        df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):
        # get the bias based on crossover
        df['curr_bias'] = np.where(df['MACD'] > df[df_col_headers.signal_header],'buy', 'sell')
        df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # slow MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines',
                       name=df_col_headers.slow_MA_header,
                       line=dict(color='red')),
            row=1, col=1)

        # fast MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines',
                       name=df_col_headers.fast_MA_header,
                       line=dict(color='green')),
            row=1, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['MACD'], mode='lines',
                       name='MACD', line=dict(color='orange')),
            row=2, col=1, secondary_y=True)

        # MACD signal line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.signal_header], mode='lines',
                       name=df_col_headers.signal_header, line=dict(color='purple')),
            row=2, col=1, secondary_y=True)

        # MACD histogram
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['MACD_hist']), row=2, col=1, secondary_y=True)

        # volume
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):
        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):
            # has a crossover occured?
            if frame['prev_bias'] and frame['curr_bias'] != frame['prev_bias']:
                return True

            #otherwise has the risk profile been broken?
            elif trade_obj.bias == 'buy' and frame['Close'] <= trade_obj.risk_price:
                return True

            elif trade_obj.bias == 'sell' and frame['Close'] >= trade_obj.risk_price:
                return True

            else:
                return False

        ###################################################################

        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame,trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame,trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

    ###########################################################
    check_MAs(params=params)

    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_MACD_V2_ABRIDGED(params, data_df, do_chart=False):
    """
    if MACD crosses above the signal line then buy

    profit exit on R multiples
    loss exit ATR risk

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'slow_MA':SLOW_MA,
                  'fast_MA':FAST_MA,
                  'signal':MACD_SIGNAL,
                  'ATR_num':ATR_NUM_RANGE,
                  #'ATR_mult':ATR_MULT_RANGE,
                  'R_mult_PT':R_PROF_TARG_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        signal_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # MACD
        df = technical_indicators.MACD(df=df, slow_MA=params['slow_MA'], fast_MA=params['fast_MA'], signal=params['signal'])
        df_col_headers.signal_header = str(params['signal'])+'signal_line'
        df_col_headers.fast_MA_header = str(params['fast_MA'])+'EMA'
        df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):
        # get the bias based on crossover
        df['curr_bias'] = np.where(df['MACD'] > df[df_col_headers.signal_header],'buy', 'sell')
        df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # slow MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines',
                       name=df_col_headers.slow_MA_header,
                       line=dict(color='red')),
            row=1, col=1)

        # fast MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines',
                       name=df_col_headers.fast_MA_header,
                       line=dict(color='green')),
            row=1, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['MACD'], mode='lines',
                       name='MACD', line=dict(color='orange')),
            row=2, col=1, secondary_y=True)

        # MACD signal line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.signal_header], mode='lines',
                       name=df_col_headers.signal_header, line=dict(color='purple')),
            row=2, col=1, secondary_y=True)

        # MACD histogram
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['MACD_hist']), row=2, col=1, secondary_y=True)

        # volume
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            elif frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'
            else:
                return False

        def exit_criteria_met(frame, trade_obj):
            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] > trade_obj.profit_target:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] < trade_obj.profit_target:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True
        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################

    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_MACD_V3_ABRIDGED(params, data_df, do_chart=False):
    """
    if MACD crosses above the signal line and both lines > 0 then buy

    profit exit on crossover
    loss exit on crossover or if ATR risk is hit

    trades will not overlap
    """

    strat_params = {'slow_MA':SLOW_MA,
                    'fast_MA':FAST_MA,
                    'signal':MACD_SIGNAL,
                    # 'ATR_mult':ATR_MULT_RANGE,
                    'ATR_num':ATR_NUM_RANGE,}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)
        return input_params_perm

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        signal_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def check_MAs(params):
        if params['fast_MA'] > params['slow_MA']:
            print('FIX YOUR MOVING AVERAGE NUMBERS')
            exit()

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # MACD
        df = technical_indicators.MACD(df=df, slow_MA=params['slow_MA'], fast_MA=params['fast_MA'], signal=params['signal'])
        df_col_headers.signal_header = str(params['signal'])+'signal_line'
        df_col_headers.fast_MA_header = str(params['fast_MA'])+'EMA'
        df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):
        # get the bias based on crossover
        df['curr_bias'] = np.where(df['MACD'] > df[df_col_headers.signal_header],'buy', 'sell')
        df['prev_bias'] = df['curr_bias'].shift(1)

        def long_short(dataframe):
            if (dataframe['MACD'] > 0) and (dataframe[df_col_headers.signal_header] > 0):
                return 'long'
            elif (dataframe['MACD'] < 0) and (dataframe[df_col_headers.signal_header] < 0):
                return 'short'
            else:
                return 'neutral'

        df['long_short'] = df.apply(long_short, axis=1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # slow MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines',
                       name=df_col_headers.slow_MA_header,
                       line=dict(color='red')),
            row=1, col=1)

        # fast MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines',
                       name=df_col_headers.fast_MA_header,
                       line=dict(color='green')),
            row=1, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['MACD'], mode='lines',
                       name='MACD', line=dict(color='orange')),
            row=2, col=1, secondary_y=True)

        # MACD signal line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.signal_header], mode='lines',
                       name=df_col_headers.signal_header, line=dict(color='purple')),
            row=2, col=1, secondary_y=True)

        # MACD histogram
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['MACD_hist']), row=2, col=1, secondary_y=True)

        # volume
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['long_short'] == 'long' and frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            if frame['long_short'] == 'short' and frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):
            # has a crossover occured?
            if frame['prev_bias'] and frame['curr_bias'] != frame['prev_bias']:
                return True

            #otherwise has the risk profile been broken?
            elif trade_obj.bias == 'buy' and frame['Close'] <= trade_obj.risk_price:
                return True

            elif trade_obj.bias == 'sell' and frame['Close'] >= trade_obj.risk_price:
                return True

            else:
                return False

        ###################################################################

        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame,trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame,trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

    ###########################################################
    check_MAs(params=params)

    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_MACD_V4_ABRIDGED(params, data_df, do_chart=False):
    """
    if MACD crosses above the signal line and both lines > 0 then buy

    profit exit on R multiples
    loss exit ATR risk

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """
    strat_params = {'slow_MA':SLOW_MA,
                    'fast_MA':FAST_MA,
                    'signal':MACD_SIGNAL,
                    'ATR_num':ATR_NUM_RANGE,
                    #'ATR_mult':ATR_MULT_RANGE,
                    'R_mult_PT':R_PROF_TARG_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        signal_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # MACD
        df = technical_indicators.MACD(df=df, slow_MA=params['slow_MA'], fast_MA=params['fast_MA'], signal=params['signal'])
        df_col_headers.signal_header = str(params['signal'])+'signal_line'
        df_col_headers.fast_MA_header = str(params['fast_MA'])+'EMA'
        df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):
        # get the bias based on crossover
        df['curr_bias'] = np.where(df['MACD'] > df[df_col_headers.signal_header],'buy', 'sell')
        df['prev_bias'] = df['curr_bias'].shift(1)

        def long_short(dataframe):
            if (dataframe['MACD'] > 0) and (dataframe[df_col_headers.signal_header] > 0):
                return 'long'
            elif (dataframe['MACD'] < 0) and (dataframe[df_col_headers.signal_header] < 0):
                return 'short'
            else:
                return 'neutral'

        df['long_short'] = df.apply(long_short, axis=1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # slow MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines',
                       name=df_col_headers.slow_MA_header,
                       line=dict(color='red')),
            row=1, col=1)

        # fast MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines',
                       name=df_col_headers.fast_MA_header,
                       line=dict(color='green')),
            row=1, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['MACD'], mode='lines',
                       name='MACD', line=dict(color='orange')),
            row=2, col=1, secondary_y=True)

        # MACD signal line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.signal_header], mode='lines',
                       name=df_col_headers.signal_header, line=dict(color='purple')),
            row=2, col=1, secondary_y=True)

        # MACD histogram
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['MACD_hist']), row=2, col=1, secondary_y=True)

        # volume
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame["long_short"] == 'long' and frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'sell':
                    return 'buy'

            elif frame["long_short"] == 'short' and frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'buy':
                    return 'sell'
            else:
                return False

        def exit_criteria_met(frame, trade_obj):
            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] > trade_obj.profit_target:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] < trade_obj.profit_target:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def MA_divergence_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    gets the X period std deviation of the difference between 2 moving averages to use as oscillator
    enters as soon as divergence breaks ranges of std deviations

    profit exit on R target
    loss exit on ATR

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'slow_MA':SLOW_MA,
                    'fast_MA': FAST_MA,
                    'use_slow_SMA':BINARY_RANGE,
                    'use_fast_SMA':BINARY_RANGE,
                    'std_dev_num':STD_DEV_RANGE,
                    'std_dev_mult':STD_DEV_MULT_RANGE,
                    'ATR_num':ATR_NUM_RANGE,
                    #'ATR_mult':ATR_MULT_RANGE,
                    'R_mult_PT':R_PROF_TARG_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        input_params_perm = adjust_for_slow_fast_MAs(input_perms=input_params_perm,
                                                     slow_idx=0,
                                                     fast_idx=1)

        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        fast_MA_header = None
        slow_MA_header = None
        std_dev_num_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # SMA/EMA
        if params['use_slow_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['slow_MA'])
            df_col_headers.slow_MA_header = str(params['slow_MA']) + 'EMA'

        if params['use_fast_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['fast_MA'])
            df_col_headers.fast_MA_header = str(params['fast_MA']) + 'EMA'

        # MA divergence
        df['MA_divergence'] = df[df_col_headers.slow_MA_header] - df[df_col_headers.fast_MA_header]

        # MA divergence std dev
        df['MA_divergence_std_dev'] = df['MA_divergence'].rolling(params['std_dev_num']).std()

        # MA divergence mean
        df['MA_divergence_mean'] = df['MA_divergence'].rolling(params['std_dev_num']).mean()

        # MA_divergence upper and lower bounds
        df['MA_divergence_upper'] = df['MA_divergence_mean'] + (df['MA_divergence_std_dev']*params['std_dev_mult'])
        df['MA_divergence_lower'] = df['MA_divergence_mean'] - (df['MA_divergence_std_dev'] * params['std_dev_mult'])

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):

        # get the long/short bias based on which side of the RSI price is on
        conditions = [df['MA_divergence'] > df['MA_divergence_upper'],
                      df['MA_divergence'] < df['MA_divergence_lower']]

        choices = ['buy', 'sell']
        df["curr_bias"] = np.select(conditions, choices, default='neutral')
        df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # slow MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.slow_MA_header], mode='lines',
                       name=df_col_headers.slow_MA_header,
                       line=dict(color='red')),
            row=1, col=1)

        # fast MA
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.fast_MA_header], mode='lines',
                       name=df_col_headers.fast_MA_header,
                       line=dict(color='green')),
            row=1, col=1)

        # MA divergence
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['MA_divergence'], mode='lines',
                       name='MA_divergence', line=dict(color='blue')),
            row=2, col=1, secondary_y=True)

        # MA divergence mean
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['MA_divergence_mean'], mode='lines',
                       name='MA_divergence_mean', line=dict(color='orange')),
            row=2, col=1, secondary_y=True)

        # MA divergence upper
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['MA_divergence_upper'], mode='lines',
                       name='MA_divergence_upper', line=dict(color='purple')),
            row=2, col=1, secondary_y=True)

        # MA divergence lower
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['MA_divergence_lower'], mode='lines',
                       name='MA_divergence_lower', line=dict(color='purple')),
            row=2, col=1, secondary_y=True)

        # volume
        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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
        ###########################################################

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'neutral':
                    return 'buy'

            elif frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'neutral':
                    return 'sell'
            else:
                return False

        def exit_criteria_met(frame, trade_obj):
            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] > trade_obj.profit_target:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] < trade_obj.profit_target:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ##################################################################################

    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def Basic_Bollinger_Bands_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    entry on break of bands
        - buy on break to downside
        - sell on break to upside

    exit either when ATR risk price hit or:
        - if buy, when close breaks upper band
        - if sell, when close touches lower band

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'BB_window':[14, 18, 26, 36, 46, 56],
                    'BB_std_dev':[1.0, 1.4, 1.8, 2.4, 3.0],
                    #'ATR_mult': [1.0, 1.5, 2.0, 2.5, 3.0]
                    'ATR_num':[15, 20, 30, 50]}

    ################################################################################
    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        BB_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(df):

        # BB
        df = technical_indicators.bollinger_bands(df=df, window=params['BB_window'], std_dev=params['BB_std_dev'])

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):

        # get the buy sell bias based on which side of upper/lower bands price is on
        conditions = [df['Close'] > df['BB upper'],
                      df['Close'] < df['BB lower']]
        choices = ['sell', 'buy']
        df["curr_bias"] = np.select(conditions, choices, default='neutral')
        df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data=[]):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        # BB mean
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB mean'], mode='lines', name='BB mean',
                       line=dict(color='orange')),
            row=1, col=1)

        # BB upper
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB upper'], mode='lines', name='BB upper',
                       line=dict(color='purple')),
            row=1, col=1)

        # BB lower
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB lower'], mode='lines', name='BB lower',
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'neutral':
                    return 'buy'

            elif frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'neutral':
                    return 'sell'
            else:
                return False

        def exit_criteria_met(frame, trade_obj):
            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] >= frame['BB upper']:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] <= frame['BB lower']:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    # check_MAs(params=params)

    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def Basic_Bollinger_Bands_V2_ABRIDGED(params, data_df, do_chart=False):
    """
    entry on break of bands
        - sell on break to downside
        - buy on break to upside

    exit either when ATR risk PT multiple is hit:

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'BB_window':list(range(10, 22, 3)) + list(range(22, 61, 5)),
                'BB_std_dev':[1.0, 1.4, 1.8, 2.4, 3.0],
                'ATR_num':list(range(15, 31, 5))+[40,50],
                #'ATR_mult':[x / 10.0 for x in range(6, 11, 4)] + [x / 10.0 for x in range(15, 26, 4)] + [3.0],
                'R_mult_PT':[1.0, 1.5, 2.0, 5.0, 7.0]}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        BB_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # BB
        df = technical_indicators.bollinger_bands(df=df, window=params['BB_window'], std_dev=params['BB_std_dev'])

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):

        # get the buy sell bias based on which side of upper/lower bands price is on
        conditions = [df['Close'] > df['BB upper'],
                      df['Close'] < df['BB lower']]
        choices = ['buy', 'sell']
        df["curr_bias"] = np.select(conditions, choices, default='neutral')
        df['prev_bias'] = df['curr_bias'].shift(1)

        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        # BB mean
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB mean'], mode='lines', name='BB mean',
                       line=dict(color='orange')),
            row=1, col=1)

        # BB upper
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB upper'], mode='lines', name='BB upper',
                       line=dict(color='purple')),
            row=1, col=1)

        # BB lower
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB lower'], mode='lines', name='BB lower',
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['curr_bias'] == 'buy' and frame['prev_bias'] == 'neutral':
                    return 'buy'

            elif frame['curr_bias'] == 'sell' and frame['prev_bias'] == 'neutral':
                    return 'sell'
            else:
                return False

        def exit_criteria_met(frame, trade_obj):

            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['High'] >= trade_obj.profit_target:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Low'] <= trade_obj.profit_target:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    # check_MAs(params=params)
    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def Basic_Bollinger_Bands_V3_ABRIDGED(params, data_df, do_chart=False):
    """
     for buys, wait for break downward through bands, take the buy when it crosses back inside
     for sells, wait for break upward through bands, take the sell when it crosses back inside

    exit either when ATR risk is hit or price breaks opposite band

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'BB_window': list(range(10, 22, 3)) + list(range(22, 61, 5)),
                    'BB_std_dev': [1.0, 1.4, 1.8, 2.4, 3.0],
                    'ATR_num': list(range(15, 31, 5)) + [40, 50],
                    #'ATR_mult': [x / 10.0 for x in range(6, 11, 4)] + [x / 10.0 for x in range(15, 26, 4)] + [3.0],
                    'R_mult_PT': [1.0, 1.5, 2.0, 5.0, 7.0]}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        BB_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # BB
        df = technical_indicators.bollinger_bands(df=df, window=params['BB_window'], std_dev=params['BB_std_dev'])

        # Band positioning
        conditions = [df['Close'] > df['BB upper'],
                      df['Close'] < df['BB lower']]
        choices = ['outside upper', 'outside lower']
        df["curr_position"] = np.select(conditions, choices, default='inside')

        df['prev_position'] = df["curr_position"].shift(1)

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        # BB mean
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB mean'], mode='lines', name='BB mean',
                       line=dict(color='orange')),
            row=1, col=1)

        # BB upper
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB upper'], mode='lines', name='BB upper',
                       line=dict(color='purple')),
            row=1, col=1)

        # BB lower
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB lower'], mode='lines', name='BB lower',
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['prev_position'] == 'outside upper' and frame['curr_position'] == 'inside':
                return 'sell'

            elif frame['prev_position'] == 'outside lower' and frame['curr_position'] == 'inside':
                return 'buy'

            else:
                return False

        def exit_criteria_met(frame, trade_obj):
            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] > frame['BB upper']:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] < frame['BB lower']:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    data_df = do_indicator_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def Intermediate_Bollinger_Bands_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    works off a 4 step series for entries

     for sells:
        price has to go from below mean (1), breaks above mean(2), breaks upper band (3), entry on break back inside upper band (4)

    for buys:
        price has to go from above mean (1), breaks below mean(2), breaks lower band (3), entry on break back inside lower band (4)

    exit either when ATR risk is hit or price breaks opposite band

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'BB_window': [10, 16, 22, 27, 32, 42],
                    'BB_std_dev': [1.0, 1.5, 2.0, 3.0],
                    'ATR_num': [15, 30, 50],
                    #'ATR_mult': [1.0, 1.5, 2.0, 3.0],
                    'R_mult_PT': [1.0, 1.5, 2.0, 3.0, 5.0]}

    ################################################################################
    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        BB_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []
        price_actions = []

    def do_indicator_calculations(df):
        """
        calculate indicators and chop data down to the slowest one
        1 = outside lower
        2 = inside lower
        3 = inside upper
        4 = outside upper
        """

        def price_band_position(dataframe):
            if dataframe['Close'] > dataframe['BB upper']:
                return 4
            elif dataframe['Close'] < dataframe['BB lower']:
                return 1
            elif dataframe['Close'] > dataframe['BB lower'] and dataframe['Close'] < dataframe['BB mean']:
                return 2
            elif dataframe['Close'] < dataframe['BB upper'] and dataframe['Close'] > dataframe['BB mean']:
                return 3

        def update_price_action_list(dataframe):

            if np.isnan(dataframe['curr_position']) and np.isnan(dataframe['prev_position']):
                return

            # if the list is blank, append the curr candle position reading to list
            if not trades_list.price_actions:
                trades_list.price_actions.append(dataframe['curr_position'])

            # otherwise if not blank then get the last updated action in the list and compare it to the current position

            else:
                # if the current position does not equal the previous position
                if dataframe['curr_position'] != trades_list.price_actions[-1]:

                    # in case price just skips over a position it will still be recorded into the price actions list
                    spaces = int(abs(dataframe['curr_position'] - dataframe['prev_position']) - 1)


                    diff = dataframe['curr_position'] - dataframe['prev_position']
                    num = 1
                    for i in range(0, spaces):
                        if diff > 0:
                            trades_list.price_actions.append(dataframe['prev_position'] + num)
                        else:
                            trades_list.price_actions.append(dataframe['prev_position'] - num)
                        num += 1

                    trades_list.price_actions.append(dataframe['curr_position'])

            return trades_list.price_actions[-4:]
        ###################################################################################

        # BB
        df = technical_indicators.bollinger_bands(df=df, window=params['BB_window'], std_dev=params['BB_std_dev'])

        # Price to Band positioning
        df["curr_position"] = df.apply(price_band_position, axis=1)
        df["prev_position"] = df['curr_position'].shift(1)

        #df[] = np.empty((len(df), 0)).tolist()
        df = df.dropna(subset=['curr_position','prev_position'])

        df['last_four'] = df.apply(update_price_action_list,axis=1)
        df["prev_last_four"] = df['last_four'].shift(1)

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        # BB mean
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB mean'], mode='lines', name='BB mean',
                       line=dict(color='orange')),
            row=1, col=1)

        # BB upper
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB upper'], mode='lines', name='BB upper',
                       line=dict(color='purple')),
            row=1, col=1)

        # BB lower
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB lower'], mode='lines', name='BB lower',
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (
                            abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (
                            abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['last_four'] != frame['prev_last_four']:

                if frame['last_four'] == [2,3,4,3]:
                    return 'sell'

                elif frame['last_four'] == [3,2,1,2]:
                    return 'buy'

            return False

        def exit_criteria_met(frame, trade_obj):
            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] > frame['BB upper']:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] < frame['BB lower']:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame, trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame, trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame, bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    # check_MAs(params=params)

    data_df = do_indicator_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def Intermediate_Bollinger_Bands_V2_ABRIDGED(params, data_df, do_chart=False):
    """
    works off a 4 step series for entries

     for sells:
        price has to go from below mean (1), breaks above mean(2), breaks upper band (3), entry on break back inside upper band (4)

    for buys:
        price has to go from above mean (1), breaks below mean(2), breaks lower band (3), entry on break back inside lower band (4)

    exit either when ATR risk is hit or R target is hit

    trades will overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'BB_window': list(range(10, 22, 3)) + list(range(22, 61, 5)),
                    'BB_std_dev': [1.0, 1.4, 1.8, 2.4, 3.0],
                    'ATR_num': list(range(15, 31, 5)) + [40, 50],
                    #'ATR_mult': [x / 10.0 for x in range(6, 11, 4)] + [x / 10.0 for x in range(15, 26, 4)] + [3.0],
                    'R_mult_PT': [1.0, 1.5, 2.0, 5.0, 7.0]}

    # strat_params = {'BB_window':FAST_MA,
    #               'BB_std_dev':STD_DEV_RANGE,
    #               'ATR_num':ATR_NUM_RANGE,
    #               'ATR_mult':ATR_MULT_RANGE,
    #               'R_mult_PT':R_PROF_TARG_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        BB_header = None
        ATR_header = None

    class trades_list:
        current = []
        completed = []
        price_actions = []

    def do_indicator_calculations(df):
        """
        calculate indicators and chop data down to the slowest one
        1 = outside lower
        2 = inside lower
        3 = inside upper
        4 = outside upper
        """

        def price_band_position(dataframe):
            if dataframe['Close'] > dataframe['BB upper']:
                return 4
            elif dataframe['Close'] < dataframe['BB lower']:
                return 1
            elif dataframe['Close'] > dataframe['BB lower'] and dataframe['Close'] < dataframe['BB mean']:
                return 2
            elif dataframe['Close'] < dataframe['BB upper'] and dataframe['Close'] > dataframe['BB mean']:
                return 3

        def update_price_action_list(dataframe):

            if np.isnan(dataframe['curr_position']) and np.isnan(dataframe['prev_position']):
                return

            # if the list is blank, append the curr candle position reading to list
            if not trades_list.price_actions:
                trades_list.price_actions.append(dataframe['curr_position'])

            # otherwise if not blank then get the last updated action in the list and compare it to the current position

            else:
                # if the current position does not equal the previous position
                if dataframe['curr_position'] != trades_list.price_actions[-1]:

                    # in case price just skips over a position it will still be recorded into the price actions list
                    spaces = int(abs(dataframe['curr_position'] - dataframe['prev_position']) - 1)
                    diff = dataframe['curr_position'] - dataframe['prev_position']
                    num = 1
                    for i in range(0, spaces):
                        if diff > 0:
                            trades_list.price_actions.append(dataframe['prev_position'] + num)
                        else:
                            trades_list.price_actions.append(dataframe['prev_position'] - num)
                        num += 1

                    trades_list.price_actions.append(dataframe['curr_position'])

            return trades_list.price_actions[-4:]
        ###################################################################################

        # BB
        df = technical_indicators.bollinger_bands(df=df, window=params['BB_window'], std_dev=params['BB_std_dev'])

        # Price to Band positioning
        df["curr_position"] = df.apply(price_band_position, axis=1)
        df["prev_position"] = df['curr_position'].shift(1)

        #df[] = np.empty((len(df), 0)).tolist()

        df['last_four'] = df.apply(update_price_action_list,axis=1)
        df["prev_last_four"] = df['last_four'].shift(1)

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        # BB mean
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB mean'], mode='lines', name='BB mean',
                       line=dict(color='orange')),
            row=1, col=1)

        # BB upper
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB upper'], mode='lines', name='BB upper',
                       line=dict(color='purple')),
            row=1, col=1)

        # BB lower
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['BB lower'], mode='lines', name='BB lower',
                       line=dict(color='purple')),
            row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (
                        abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (
                        abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['last_four'] != frame['prev_last_four']:

                if frame['last_four'] == [2, 3, 4, 3]:
                    return 'sell'

                elif frame['last_four'] == [3, 2, 1, 2]:
                    return 'buy'

            return False

        def exit_criteria_met(frame, trade_obj):

            def loss_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def profit_exit_hit():
                if trade_obj.bias == 'buy':
                    if frame['Close'] > trade_obj.profit_target:
                        return True

                if trade_obj.bias == 'sell':
                    if frame['Close'] < trade_obj.profit_target:
                        return True

                return False

            ########################################

            if loss_exit_hit() or profit_exit_hit():
                return True

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame, trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame, trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame, bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################
    # check_MAs(params=params)
    data_df = do_indicator_calculations(df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_heiken_ashi_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    entries:
    buy signal on first green candle after series of x consecutive red candle
    sell signal on first red candle after series of x consecutive green candle

    exit/reverse on either close of candle that breaks ATR risk price or close of first opposite candle

    trades will not overlap
    """

    strat_params = {'consec_candles':[3, 4, 5, 6, 8, 10, 12, 14],
                    #'ATR_mult': [0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0]
                    'ATR_num':[15, 21, 27, 33, 39, 42, 48, 54]}

    ################################################################################
    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        ATR_header = None

    class trades_list:
        current = []
        completed = []

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""
        # candle_num
        data_df = data_df.reset_index()

        # heiken ashi
        data_df = technical_indicators.heiken_ashi(df=data_df)
        data_df['prev_HA_red_green'] = data_df['HA_red_green'].shift(1)
        data_df['is_pivot_point'] = np.where(data_df['prev_HA_red_green'] != data_df['HA_red_green'], 1,0)

        data_df["consec_red_candles"] = data_df.groupby((data_df["prev_HA_red_green"] == 'green').cumsum()).cumcount()
        data_df["consec_green_candles"] = data_df.groupby((data_df["prev_HA_red_green"] == 'red').cumsum()).cumcount()

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['HA Open'], high=data_df['HA High'],
                                     low=data_df['HA Low'], close=data_df['HA Close']), row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['HA Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['HA Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):
            if frame['is_pivot_point'] == 1:
                if frame['HA_red_green'] == 'red' and frame["consec_green_candles"] >= params['consec_candles']:
                    return 'sell'
                if frame['HA_red_green'] == 'green' and frame["consec_red_candles"] >= params['consec_candles']:
                    return 'buy'

            return False

        def exit_criteria_met(frame, trade_obj):

            # exit on candle color flip
            if frame['HA_red_green'] != frame['prev_HA_red_green']:
                return True

            # exit on risk price
            if trade_obj.bias == 'buy' and frame['HA Close'] <= trade_obj.risk_price:
                return True
            if trade_obj.bias == 'sell' and frame['HA Close'] >= trade_obj.risk_price:
                return True

            return False

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame, trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame, trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_heiken_ashi_V2_ABRIDGED(params, data_df, do_chart=False):
    """
    entries:
    buy signal on first green candle after series of x consecutive red candle
    sell signal on first red candle after series of x consecutive green candle

    - however avg distance from first color change point to curr color change point needs to be > y ATR of the first color change
    - avg distance = abs val of avg(OHLC) of p0 to avg(OHLC) of p1

    exit/reverse on either ATR risk price or close of first opposite candle

    trades will not overlap
    """

    strat_params = {'consec_candles':[3, 4, 6, 8, 10, 12, 14],
                  'ATR_num':[15, 20, 25, 30, 40, 50],
                  #'ATR_mult':[0.6, 1.0, 1.5, 2.0, 3.0],
                  'ATR_PP_chg_mult':[1.0, 1.5, 2.0, 3.0, 5.0]} # the multiple of ATR used to determine if two pivot points are far enough apart}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        ATR_header = None

    class trades_list:
        current = []
        completed = []
        prev_avg_HA_OHLC = [] #candle color, high/low price depending on candle color

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""
        # candle_num
        data_df = data_df.reset_index()

        # heiken ashi
        data_df = technical_indicators.heiken_ashi(df=data_df)

        data_df['prev_HA_red_green'] = data_df['HA_red_green'].shift(1)
        data_df['is_pivot_point'] = np.where(data_df['prev_HA_red_green'] != data_df['HA_red_green'], 1,0)

        data_df['pivot_price'] = np.where(data_df['HA_red_green'] == 'green', data_df['HA Open'],data_df['HA Close'])

        data_df["consec_red_candles"] = data_df.groupby((data_df["prev_HA_red_green"] == 'green').cumsum()).cumcount()
        data_df["consec_green_candles"] = data_df.groupby((data_df["prev_HA_red_green"] == 'red').cumsum()).cumcount()

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['HA Open'], high=data_df['HA High'],
                                     low=data_df['HA Low'], close=data_df['HA Close']), row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['HA Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['HA Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            def potential_entry(frame):

                if frame['is_pivot_point'] == 1:
                    if frame['HA_red_green'] == 'red' and (frame["consec_green_candles"] >= params['consec_candles']):
                        return True
                    elif frame['HA_red_green'] == 'green' and (frame["consec_red_candles"] >= params['consec_candles']):
                        return True

                return False

            def check_volatilty(frame):
                cutoff = frame[df_col_headers.ATR_header] * params['ATR_PP_chg_mult']
                pivot_difference = abs(frame['pivot_price'] - trades_list.prev_avg_HA_OHLC[1])
                #print('     ATR cutoff:',cutoff)
                #print(      trades_list.prev_avg_HA_OHLC)
                #print('     pivot diffs:',pivot_difference)

                if pivot_difference > cutoff:
                    #print('     pivot diff > cutoff...')
                    return True

                return False

            ################################################################################

            # if this is a pivot point and theres been x consecutive same color candles
            if potential_entry(frame=frame):
                #print(frame['Date_Time'])
                #print('is pivot point')

                if check_volatilty(frame=frame):
                    if frame['HA_red_green'] == 'red':
                        #print('             ENTERING SELL POSITION')
                        #print('-----------------------------------------------------------------------')
                        return 'sell'

                    if frame['HA_red_green'] == 'green':
                        #print('         ENTERING BUY POSITION')
                        #print('-----------------------------------------------------------------------')
                        return 'buy'

            #print('-----------------------------------------------------------------------')
            return False

        def exit_criteria_met(frame, trade_obj):

            # exit on candle color flip
            if frame['HA_red_green'] != frame['prev_HA_red_green']:
                return True

            # exit on risk price
            if trade_obj.bias == 'buy' and frame['HA Close'] <= trade_obj.risk_price:
                return True
            if trade_obj.bias == 'sell' and frame['HA Close'] >= trade_obj.risk_price:
                return True

            return False

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        # are you currently in a trade? (yes)

        if not trades_list.prev_avg_HA_OHLC and frame['is_pivot_point'] == 1:
            trades_list.prev_avg_HA_OHLC = [frame['HA_red_green'], frame['pivot_price']]
            return

        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame, trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame, trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if not trades_list.current:

            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

        if frame['is_pivot_point'] == 1:
            trades_list.prev_avg_HA_OHLC = [frame['HA_red_green'], frame['pivot_price']]

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_heiken_ashi_V3_ABRIDGED(params, data_df, do_chart=False):
    """
    entries:
    buy signal on first green candle after series of x consecutive red candle
    sell signal on first red candle after series of x consecutive green candle

    - however avg distance from first color change point to curr color change point needs to be > y ATR of the first color change
    - avg distance = opening(red) - closing(green) difference between p0 to avg(OHLC) of p1

    exit/reverse on either ATR risk price or if PT hit

    trades will overlap
    """

    strat_params = {'consec_candles':[4, 5, 6, 8, 10, 12, 14],
                  'ATR_num':[15, 30, 50],
                  #'ATR_mult':[1.0, 1.5, 2.0, 3.0], # ATR_risk_mult
                  'ATR_PP_chg_mult':[1.0, 1.5, 2.0, 3.0], # the multiple of ATR used to determine if two pivot points are far enough apart
                  'R_mult_PT':[1.0, 1.5, 2.0, 5.0]}

    ################################################################################
    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)

        return input_params_perm, labels

    ##################################################################################

    class df_col_headers:
        ATR_header = None

    class trades_list:
        current = []
        completed = []
        prev_avg_HA_OHLC = [] #candle color, high/low price depending on candle color

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""
        # candle_num
        data_df = data_df.reset_index()

        # heiken ashi
        data_df = technical_indicators.heiken_ashi(df=data_df)

        data_df['prev_HA_red_green'] = data_df['HA_red_green'].shift(1)
        data_df['is_pivot_point'] = np.where(data_df['prev_HA_red_green'] != data_df['HA_red_green'], 1,0)

        data_df['pivot_price'] = np.where(data_df['HA_red_green'] == 'green', data_df['HA Open'],data_df['HA Close'])

        data_df["consec_red_candles"] = data_df.groupby((data_df["prev_HA_red_green"] == 'green').cumsum()).cumcount()
        data_df["consec_green_candles"] = data_df.groupby((data_df["prev_HA_red_green"] == 'red').cumsum()).cumcount()

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['HA Open'], high=data_df['HA High'],
                                     low=data_df['HA Low'], close=data_df['HA Close']), row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['HA Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['HA Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (params['R_mult_PT'] * frame[df_col_headers.ATR_header])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (params['R_mult_PT'] * frame[df_col_headers.ATR_header])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            def potential_entry(frame):

                if frame['is_pivot_point'] == 1:
                    if frame['HA_red_green'] == 'red' and (frame["consec_green_candles"] >= params['consec_candles']):
                        return True
                    elif frame['HA_red_green'] == 'green' and (frame["consec_red_candles"] >= params['consec_candles']):
                        return True

                return False

            def check_volatilty(frame):
                cutoff = frame[df_col_headers.ATR_header] * params['ATR_PP_chg_mult']
                pivot_difference = abs(frame['pivot_price'] - trades_list.prev_avg_HA_OHLC[1])
                #print('     ATR cutoff:',cutoff)
                #print(      trades_list.prev_avg_HA_OHLC)
                #print('     pivot diffs:',pivot_difference)

                if pivot_difference > cutoff:
                    #print('     pivot diff > cutoff...')
                    return True

                return False

            ################################################################################

            # if this is a pivot point and theres been x consecutive same color candles
            if potential_entry(frame=frame):
                #print(frame['Date_Time'])
                #print('is pivot point')

                if check_volatilty(frame=frame):
                    if frame['HA_red_green'] == 'red':
                        #print('             ENTERING SELL POSITION')
                        #print('-----------------------------------------------------------------------')
                        return 'sell'

                    if frame['HA_red_green'] == 'green':
                        #print('         ENTERING BUY POSITION')
                        #print('-----------------------------------------------------------------------')
                        return 'buy'

            #print('-----------------------------------------------------------------------')
            return False

        def exit_criteria_met(frame, trade_obj):

            if trade_obj.bias == 'buy':
                if frame['HA Close'] >= trade_obj.profit_target or frame['HA Close'] <= trade_obj.risk_price:
                    return True

            elif trade_obj.bias == 'sell':
                if frame['HA Close'] <= trade_obj.profit_target or frame['HA Close'] >= trade_obj.risk_price:
                    return True

            return False

        ###################################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        # are you currently in a trade? (yes)

        if not trades_list.prev_avg_HA_OHLC and frame['is_pivot_point'] == 1:
            trades_list.prev_avg_HA_OHLC = [frame['HA_red_green'], frame['pivot_price']]
            return

        # are you currently in a trade? (yes)
        if trades_list.current:
            curr_trade = trades_list.current[0]
            # has the exit criteria for that trade been met? (yes)
            if exit_criteria_met(frame=frame, trade_obj=curr_trade):
                # complete the trade and exit
                curr_trade = exit_trade(frame=frame, trade_obj=curr_trade)

                # record the trade
                record_trade(trade_obj=curr_trade)

                # delete the trade from current trades list
                trades_list.current.remove(curr_trade)

        # are you currently in a trade? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met?
            trade_bias = entry_criteria_met(frame=frame)
            # has an entry been triggered? (yes)
            if trade_bias:
                # enter on the close of the breaking candle
                curr_trade = enter_trade(frame=frame, bias=trade_bias)
                trades_list.current.append(curr_trade)

        if frame['is_pivot_point'] == 1:
            trades_list.prev_avg_HA_OHLC = [frame['HA_red_green'], frame['pivot_price']]

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def basic_ichimoku_V1_ABRIDGED(params, data_df, do_chart=False):
    """
    enter long when
        - price closes above projected cloud
        - current cloud is green
        - conversion line is above base line
        - lagging line above lagging line projected cloud

    enter short on opposite

    does use the lagging line (chikou)

    based risk/judge reward on atr

    trades may overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'ATR_num': ATR_NUM_RANGE,
                    #'ATR_mult': ATR_MULT_RANGE,
                    'ATR_PP_chg_mult': ATR_MULT_RANGE,# the multiple of ATR used to determine if two pivot points are far enough apart
                    'R_mult_PT': R_PROF_TARG_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        labels = create_permuations_labels(params_dict=strat_params)
        offset_candles = max([max(item) for item in strat_params.values()])

        return input_params_perm, labels, offset_candles

    ##################################################################################

    class trades_list:
        current = []
        completed = []

    class df_col_headers:
        ATR_header = None

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""

        def get_price_to_cloud_bias(dataframe):
            if dataframe['Close'] > dataframe['projected_cloud_max']:
                return 'long'
            elif dataframe['Close'] < dataframe['projected_cloud_min']:
                return 'short'
            else:
                return 'None'

        def get_curr_cloud_bias(dataframe):
            if dataframe["Current Senkou A"] > dataframe["Current Senkou B"]:
                return 'long'
            elif dataframe["Current Senkou A"] < dataframe["Current Senkou B"]:
                return 'short'
            else:
                return 'None'

        def get_conversion_to_base_bias(dataframe):
            if dataframe['Tenkan'] > dataframe['Kijun']:
                return 'long'
            elif dataframe['Tenkan'] <= dataframe['Kijun']:
                return 'short'

        def get_chikou_bias(dataframe):
            if dataframe['Chikou'] > dataframe['chikou_cloud_max']:
                return 'long'
            elif dataframe['Chikou'] < dataframe['chikou_cloud_min']:
                return 'short'
            else:
                return 'None'

        def get_buy_sell_bias(dataframe):
            if dataframe['bias_1'] == 'long' and dataframe['bias_2'] == 'long' and dataframe['bias_3'] == 'long' and \
                    dataframe['bias_4'] == 'long':
                return 'buy'
            elif dataframe['bias_1'] == 'short' and dataframe['bias_2'] == 'short' and dataframe[
                'bias_3'] == 'short' and dataframe['bias_4'] == 'short':
                return 'sell'
            else:
                return 'no_bias'

        ####################################################################################
        # candle_num
        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # ichimoku
        data_df = technical_indicators.ichimoku(df=data_df)

        data_df['projected_cloud_max'] = data_df[["Projected Senkou A", "Projected Senkou B"]].max(axis=1)
        data_df['projected_cloud_min'] = data_df[["Projected Senkou A", "Projected Senkou B"]].min(axis=1)

        data_df['chikou_cloud_max'] = data_df[['Chikou adj Senkou B', 'Chikou adj Senkou A']].max(axis=1)
        data_df['chikou_cloud_min'] = data_df[['Chikou adj Senkou B', 'Chikou adj Senkou A']].min(axis=1)

        data_df['bias_1'] = data_df.apply(get_price_to_cloud_bias, axis=1)
        data_df['bias_2'] = data_df.apply(get_curr_cloud_bias, axis=1)
        data_df['bias_3'] = data_df.apply(get_conversion_to_base_bias, axis=1)
        data_df['bias_4'] = data_df.apply(get_chikou_bias, axis=1)

        data_df['curr_bias'] = data_df.apply(get_buy_sell_bias, axis=1)

        data_df = data_df.dropna()

        return data_df

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # conversion line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Tenkan'], mode='lines', name='conversion line',
                       line=dict(color='orange')), row=1, col=1)

        # base line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Kijun'], mode='lines', name='base line',
                       line=dict(color='blue')), row=1, col=1)

        # leading span A
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Projected Senkou A'], mode='lines', name='leading span A',
                       line=dict(color='green')), row=1, col=1)

        # leading span B
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Projected Senkou B'], mode='lines', name='leading span B',
                       line=dict(color='red')), row=1, col=1)

        # chikou
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Chikou'], mode='lines', name='Chikou',
                       line=dict(color='purple')), row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)

        fig.show()

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            # get all original variable names
            accepted_params_list = [attr for attr in dir(trade_metrics()) if
                                    not callable(getattr(trade_metrics(), attr)) and not attr.startswith("__")]

            asdf = vars(trade_obj)
            for k, v in asdf.items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame, trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if curr_trade.bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
            elif curr_trade.bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            if frame['curr_bias'] != frame['prev_bias']:

                if frame['curr_bias'] == 'buy':
                    return 'buy'

                if frame['curr_bias'] == 'sell':
                    return 'sell'

            return False

        def exit_criteria_met(frame, trade_obj):

            def loss_exit_hit():  # is ATR risk price
                if trade_obj.bias == 'buy':
                    if frame['Close'] < trade_obj.risk_price:
                        return True
                if trade_obj.bias == 'sell':
                    if frame['Close'] > trade_obj.risk_price:
                        return True

                return False

            def target_exit_hit():
                if trade_obj.bias == 'buy':
                    PT = trade_obj.entry_price + (
                                abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                    if frame['Close'] > PT:
                        return True

                if trade_obj.bias == 'sell':
                    PT = trade_obj.entry_price - (
                                abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                    if frame['Close'] < PT:
                        return True

                return False

            ########################################

            if loss_exit_hit() or target_exit_hit():
                return True

        ############################################################

        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame, trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame, trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame, bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    data_df['prev_bias'] = data_df['curr_bias'].shift(1)
    data_df = data_df.iloc[1:]

    data_df.apply(get_trades, axis=1)

    ########################################################

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed


# IS BASICALLY JUST A BOLLINGER BAND STRATEGY
################################################################################################
def X_MA_crossover_mean_reversion_V2(params, data_df, do_chart=False):
    """
    simple 1 moving average crossover strategy

    enter long on candle close when price is greater than x std dev from the MA
    emter short on candle close when price is less than x std dev from the MA

    loss exit on ATR multiple
    profit exit on a R levels

    base risk/judge reward off X atr multiple

    trades will overlap (limit or 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times

    """

    if params == 'test':
        params = {'MA':SINGLE_MA_RANGE,
                  'use_SMA':BINARY_RANGE,
                  'std_dev_num':STD_DEV_RANGE,
                  'std_dev_mult':STD_DEV_MULT_RANGE,
                  'ATR_num':ATR_NUM_RANGE,
                  #'ATR_mult':ATR_MULT_RANGE,
                  'R_mult_PT':R_PROF_TARG_RANGE}

    class df_col_headers:
        MA_header = None
        ATR_header = None

    def do_indicator_calculations(df):
        """calculate indicators and chop data down to the slowest one"""

        # SMA/EMA
        if params['use_SMA'] == 1:
            df = technical_indicators.simple_moving_average(df=df, MA_num=params['MA'])
            df_col_headers.MA_header = str(params['MA']) + 'SMA'
        else:
            df = technical_indicators.exponential_moving_average(df=df, MA_num=params['MA'])
            df_col_headers.MA_header = str(params['MA']) + 'EMA'

        # MA to close diff
        df['MA_close_diff'] = df['Close'] - df[df_col_headers.MA_header]

        # std dev
        df['std_dev x mult'] = ((df[df_col_headers.MA_header] - df['Close']).rolling(
            params['std_dev_num']).std()) * params['std_dev_mult']

        # ATR
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

        df = df.dropna(subset=indicator_col_names)
        df = df.reset_index(drop=True)
        return df

    def do_bias_calculations(df):

        def buy_sell(dataframe):
            if dataframe['MA_close_diff'] < 0 and abs(dataframe['MA_close_diff']) > dataframe['std_dev x mult'] or dataframe['MA_close_diff'] < 0:
                return 'buy'
            elif dataframe['MA_close_diff'] >= 0 and abs(dataframe['MA_close_diff']) > dataframe['std_dev x mult'] or dataframe['MA_close_diff'] > 0:
                return 'sell'



        df['bias'] = df.apply(buy_sell, axis=1)

        return df

    def enter_trade(trade_num):
        curr_trade.trade_num = trade_num
        curr_trade.bias = curr_bias
        curr_trade.entry_datetime = curr_candle['Date_Time']
        curr_trade.entry_price = curr_candle['Close']

        if curr_bias == 'buy':
            curr_trade.risk_price = curr_trade.entry_price - curr_candle[df_col_headers.ATR_header]
        else:
            curr_trade.risk_price = curr_trade.entry_price + curr_candle[df_col_headers.ATR_header]

    def exit_trade(trade_obj):
        trade_obj.exit_price = curr_candle['Close']
        trade_obj.exit_datetime = curr_candle['Date_Time']
        return trade_obj

    def exit_criteria_met(trade_obj):
        def loss_exit_hit():
            if trade_obj.bias == 'buy':
                if curr_candle['Close'] < trade_obj.risk_price:
                    return True
            if trade_obj.bias == 'sell':
                if curr_candle['Close'] > trade_obj.risk_price:
                    return True

            return False

        def profit_exit_hit():
            if trade_obj.bias == 'buy':
                PT = trade_obj.entry_price + (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] > PT:
                    return True

            if trade_obj.bias == 'sell':
                PT = trade_obj.entry_price - (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] < PT:
                    return True

            return False

        ########################################

        if loss_exit_hit():
            return True
        elif profit_exit_hit():
            return True

    def record_trade(trade_obj):

        """
        convert class instance variables to dict, append dict to trades_list
        """

        trade_dict = {}

        for k, v in vars(trade_obj).items():
            if k.startswith('_'):
                continue
            trade_dict[k] = v

        completed_trades_list.append(trade_dict)

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.MA_header], mode='lines',
                       name=df_col_headers.MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['std_dev x mult'], mode='lines',
                       name='+ std dev x mult',
                       line=dict(color='purple')),
            row=2, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=-(data_df['std_dev x mult']), mode='lines',
                       name='- std dev x mult',
                       line=dict(color='purple')),
            row=2, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=(data_df['Close'] - data_df[df_col_headers.MA_header]),
                       mode='lines',
                       name='Close to MA difference',
                       line=dict(color='green')),
            row=2, col=1)

        # fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    ###########################################################
    # check_MAs(params=params)

    data_df = do_indicator_calculations(df=data_df)
    data_df = do_bias_calculations(df=data_df)

    uncompleted_trades_list = []
    completed_trades_list = []

    # call the class instance
    curr_trade = trade_metrics()

    prev_bias = None
    curr_bias = None
    trade_num = 0

    # loop through df
    for idx in data_df.index:

        # get the current candle data and bias
        curr_candle = data_df.loc[idx]
        curr_bias = curr_candle['bias']

        # if previous bias does not exist, set it equal to current bias and then continue
        # i.e if its the first iteration
        if not prev_bias:
            prev_bias = curr_bias
            continue

        # do you have any ongoing trades?
        if uncompleted_trades_list:
            # creat a blank temp list to store trades
            temp_list = []

            for trade in uncompleted_trades_list:
                if exit_criteria_met(trade_obj=trade):
                    trade = exit_trade(trade_obj=trade)
                    record_trade(trade_obj=trade)

                else:
                    temp_list.append(trade)

            uncompleted_trades_list = temp_list

        # are there any new trades?
        # make sure you're not already in more than 3 trades
        if len(uncompleted_trades_list) < 3:
            # has a crossover occured?
            if prev_bias != curr_bias:
                enter_trade(trade_num)
                trade_num += 1

                uncompleted_trades_list.append(curr_trade)
                curr_trade = trade_metrics()

        prev_bias = curr_bias

    if do_chart:
        view_chart(trade_data=completed_trades_list)

    return completed_trades_list

def X_MA_crossover_mean_reversion_V3(params, data_df, do_chart=False):
    """
    simple 1 moving average crossover strategy

    enter long on candle close when price is greater than x std dev from the MA
    emter short on candle close when price is less than x std dev from the MA

    loss exit on ATR multiple
    profit exit on a R levels

    base risk/judge reward off X atr multiple

    trades will overlap (limit or 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times

    *** calculations arent right but its kind of a fluke, going to keep it
    """

    if params == 'test':
        params = {'MA':SINGLE_MA_RANGE,
                  'use_SMA':BINARY_RANGE,
                  'std_dev_num':STD_DEV_RANGE,
                  'std_dev_mult':STD_DEV_MULT_RANGE,
                  'ATR_num':ATR_NUM_RANGE,
                  #'ATR_mult':ATR_MULT_RANGE,
                  'R_mult_PT':R_PROF_TARG_RANGE}

    class df_col_headers:
        MA_header = None
        ATR_header = None

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""

        # SMA/EMA
        if params['use_SMA'] == 1:
            data_df = technical_indicators.simple_moving_average(df=data_df, MA_num=params['MA'])
            df_col_headers.MA_header = str(params['MA']) + 'SMA'
        else:
            data_df = technical_indicators.exponential_moving_average(df=data_df, MA_num=params['MA'])
            df_col_headers.MA_header = str(params['MA']) + 'EMA'

        # std deviation of diff between MA and price close x the multiple
        data_df['std_dev'] = (abs(data_df[df_col_headers.MA_header] - data_df['Close']).rolling(params['std_dev_num']).std())*params['std_dev_mult']

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def enter_trade(trade_num):
        curr_trade.trade_num = trade_num
        curr_trade.bias = curr_bias
        curr_trade.entry_datetime = curr_candle['Date_Time']
        curr_trade.entry_price = curr_candle['Close']

        if curr_bias == 'buy':
            curr_trade.risk_price = curr_trade.entry_price - curr_candle[df_col_headers.ATR_header]
        else:
            curr_trade.risk_price = curr_trade.entry_price + curr_candle[df_col_headers.ATR_header]

    def exit_trade(trade_obj):
        trade_obj.exit_price = curr_candle['Close']
        trade_obj.exit_datetime = curr_candle['Date_Time']
        return trade_obj

    def exit_criteria_met(trade_obj):
        def loss_exit_hit():
            if trade_obj.bias == 'buy':
                if curr_candle['Close'] < trade_obj.risk_price:
                    return True
            if trade_obj.bias == 'sell':
                if curr_candle['Close'] > trade_obj.risk_price:
                    return True

            return False

        def profit_exit_hit():
            if trade_obj.bias == 'buy':
                PT = trade_obj.entry_price + (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] > PT:
                    return True

            if trade_obj.bias == 'sell':
                PT = trade_obj.entry_price - (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] < PT:
                    return True

            return False

        ########################################

        if loss_exit_hit():
            return True
        elif profit_exit_hit():
            return True

    def record_trade(trade_obj):

        """
        convert class instance variables to dict, append dict to trades_list
        """

        trade_dict = {}

        for k, v in vars(trade_obj).items():
            if k.startswith('_'):
                continue
            trade_dict[k] = v

        completed_trades_list.append(trade_dict)

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df[df_col_headers.MA_header], mode='lines', name=df_col_headers.MA_header,
                       line=dict(color='orange')),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['std_dev'], mode='lines',
                       name='std dev x mult',
                       line=dict(color='purple')),
            row=2, col=1)

        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=abs(data_df['Close'] - data_df[df_col_headers.MA_header]), mode='lines',
                       name='Close to MA difference',
                       line=dict(color='green')),
            row=2, col=1)

        #fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    ###########################################################
    # check_MAs(params=params)

    data_df = do_indicator_calculations(data_df=data_df)

    # get the bias
    data_df['bias'] = np.where(abs(data_df['Close']-data_df[df_col_headers.MA_header]) > data_df['std_dev'], 'buy', 'sell')

    uncompleted_trades_list = []
    completed_trades_list = []

    # call the class instance
    curr_trade = trade_metrics()

    prev_bias = None
    curr_bias = None
    trade_num = 0

    # loop through df
    for idx in data_df.index:

        # get the current candle data and bias
        curr_candle = data_df.loc[idx]
        curr_bias = curr_candle['bias']

        # if previous bias does not exist, set it equal to current bias and then continue
        # i.e if its the first iteration
        if not prev_bias:
            prev_bias = curr_bias
            continue

        # do you have any ongoing trades?
        if uncompleted_trades_list:
            # creat a blank temp list to store trades
            temp_list = []

            for trade in uncompleted_trades_list:
                if exit_criteria_met(trade_obj=trade):
                    trade = exit_trade(trade_obj=trade)
                    record_trade(trade_obj=trade)

                else:
                    temp_list.append(trade)

            uncompleted_trades_list = temp_list

        # are there any new trades?
        # make sure you're not already in more than 3 trades
        if len(uncompleted_trades_list) < 3:
            # has a crossover occured?
            if prev_bias != curr_bias:
                enter_trade(trade_num)
                trade_num += 1

                uncompleted_trades_list.append(curr_trade)
                curr_trade = trade_metrics()


        prev_bias = curr_bias

    if do_chart:
        view_chart(trade_data=completed_trades_list)

    return completed_trades_list
################################################################################################


# VVVVVVVVVVVVVVVVVVVVVVV NOT DONE VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

def basic_heiken_ashi_V4(params, data_df, do_chart=False):
    """
    entry params:
    - buy entry signal is first candle to make a new high (entry price = price of previous high)
    - sell entry signal is first candle to make a new low (entry price = price of previous low)
    however
    - distance between last pivot point entry signal must be >= params.consec candles
    - distance from last pivot point avg OHLC to curr candle avg OHLC signal needs to be >= y (ATR x params.ATR_PP_chg_mult)

    exit/reverse on either close of candle that breaks ATR risk price or on high/low price of first candle to make new low (buys)/ high(sells)

    trades may overlap
    """

    if params == 'test':
        params = {'consec_candles':HEIKEN_ASHI_RANGE,
                  'ATR_num':ATR_NUM_RANGE,
                  #'ATR_mult':ATR_MULT_RANGE,
                  'ATR_PP_chg_mult':ATR_MULT_RANGE} # the multiple of ATR used to determine if two pivot points are far enough apart}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        return input_params_perm

    ##################################################################################

    class df_col_headers:
        ATR_header = None

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""
        # candle_num
        data_df = data_df.reset_index()

        # heiken ashi
        data_df = technical_indicators.heiken_ashi(df=data_df)
        data_df['avg HA OHLC'] = (data_df['HA Open'] + data_df['HA High'] + data_df['HA Low'] + data_df['HA Close'])/4

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def enter_trade(trade_num):
        curr_trade.trade_num = trade_num
        curr_trade.entry_datetime = curr_candle['Date_Time']
        curr_trade.entry_price = previous_candle['HA High'] # entering on the break of the previous candle high

        if curr_trade.bias == 'buy':
            curr_trade.risk_price = curr_trade.entry_price - curr_candle[df_col_headers.ATR_header]
        if curr_trade.bias == 'sell':
            curr_trade.risk_price = curr_trade.entry_price + curr_candle[df_col_headers.ATR_header]

    def exit_trade(trade_obj):
        if trade_obj.bias == 'buy':
            trade_obj.exit_price = previous_candle['HA Low']
        if trade_obj.bias == 'sell':
            trade_obj.exit_price = previous_candle['HA High']

        trade_obj.exit_datetime = curr_candle['Date_Time']

    def exit_criteria_met(trade_obj):

        if trade_obj.bias == 'buy':
            if curr_candle['HA Close'] <= trade_obj.risk_price or curr_candle['HA Low'] < previous_candle['HA Low']:
                # exit on risk price or if the low of the current candle is < previous candle low
                return True

        if trade_obj.bias == 'sell':
            if curr_candle['HA Close'] >= trade_obj.risk_price or curr_candle['HA High'] > previous_candle['HA High']:
                # exit on risk price or if the high of the current candle is > previous candle high
                return True

        return False

    def record_trade(trade_obj):

        """
        convert class instance variables to dict, append dict to trades_list
        """

        trade_dict = {}

        # get all original variable names
        accepted_params_list = [attr for attr in dir(trade_metrics()) if not callable(getattr(trade_metrics(), attr)) and not attr.startswith("__")]

        for k, v in vars(trade_obj).items():
            if k.startswith('_'):
                continue
            if k not in accepted_params_list:
                continue

            trade_dict[k] = v

        completed_trades_list.append(trade_dict)

    def test_for_pivot_point():
        if curr_candle['HA_red_green'] != previous_candle['HA_red_green']:
            return True

    def is_dup_trade(input_list,trade_obj):
        # tests if a trade already exists in the list of objects
        if input_list:
            if any(trade_obj.__dict__ == i.__dict__ for i in input_list):
                return True

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['HA Open'], high=data_df['HA High'],
                                     low=data_df['HA Low'], close=data_df['HA Close']), row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    # call the class instance
    curr_trade = trade_metrics()

    uncompleted_trades_list = []
    completed_trades_list = []
    pivot_point_list = []

    previous_candle = None
    trade_num = 0

    # loop through df
    for idx in data_df.index:

        # get the current candle data
        curr_candle = data_df.loc[idx]

        # if its the first iteration
        if (idx - 1) < 0:
            continue
        previous_candle = data_df.loc[idx - 1]

        # make sure at least the amount of candles in consec params have gone by
        if (idx - (params['consec_candles'] + 1)) < 0:
            if test_for_pivot_point():
                pivot_point_list.append(curr_candle)
            continue

        # are you in trades
        if uncompleted_trades_list:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through trades
            for trade in uncompleted_trades_list:
                # test for exit criteria
                if exit_criteria_met(trade_obj=trade):
                    exit_trade(trade_obj=trade)
                    record_trade(trade_obj=trade)

                else:
                    temp_list.append(trade)

            uncompleted_trades_list = temp_list


        # look for more trade entries
        # is your entry criteria met?

        last_x_candles = data_df.iloc[(idx - params['consec_candles']):idx] # excludes current candle
        last_x_candle_colors = list(last_x_candles['HA_red_green'])

        if len(pivot_point_list) >= 1:
            # make sure candles are at least x candles apart
            if (curr_candle['index'] - pivot_point_list[-1]['index']) >= params['consec_candles']:
                # make sure they are far enough apart price wise
                if abs(pivot_point_list[-1]['avg HA OHLC'] - curr_candle['avg HA OHLC']) > (pivot_point_list[-1][df_col_headers.ATR_header] * params['ATR_PP_chg_mult']):

                    # if last x candles are all red and curr candle high is greater than prev candle high
                    if (all(x == 'red' for x in last_x_candle_colors)) and (curr_candle['HA High'] > previous_candle['HA High']): # is high of [current candle] > last candle high
                        # make sure is not dup trade before adding to uncompleted trade list
                        if not is_dup_trade(input_list=uncompleted_trades_list,trade_obj=curr_trade):
                            curr_trade.bias = 'buy' #set the bias
                            enter_trade(trade_num)  #enter the trade
                            trade_num += 1          #increase the trade number
                            uncompleted_trades_list.append(curr_trade) #append to uncomp trade list
                            curr_trade = trade_metrics()    #reset the curr trade class object

                    # if last x candles are all green and curr candle low is less than prev candle high
                    if (all(x == 'green' for x in last_x_candle_colors)) and (curr_candle['HA Low'] < previous_candle['HA Low']): # is low of [current candle] < last candle low
                        # make sure is not dup trade before adding to uncompleted trade list
                        if not is_dup_trade(input_list=uncompleted_trades_list, trade_obj=curr_trade):
                            curr_trade.bias = 'sell' #set the bias
                            enter_trade(trade_num)   #enter the trade
                            trade_num += 1           #increase the trade number
                            uncompleted_trades_list.append(curr_trade)  #append to uncomp trade list
                            curr_trade = trade_metrics()    #reset the curr trade class object

        if test_for_pivot_point():
            pivot_point_list.append(curr_candle)

    if do_chart:
        view_chart(trade_data=completed_trades_list)

    return completed_trades_list

def basic_heiken_ashi_V4_ABRIDGED(params, data_df, do_chart=False):
    """
    entry params:
    - buy entry signal is first candle to make a new high (entry price = price of previous high)
    - sell entry signal is first candle to make a new low (entry price = price of previous low)
    however
    - distance between last pivot point entry signal must be >= params.consec candles
    - distance from last pivot point avg OHLC to curr candle avg OHLC signal needs to be >= y (ATR x params.ATR_PP_chg_mult)

    exit/reverse on either close of candle that breaks ATR risk price or on high/low price of first candle to make new low (buys)/ high(sells)

    trades may overlap


    """

    if params == 'test':
        params = {'consec_candles': HEIKEN_ASHI_RANGE,
                  'ATR_num': ATR_NUM_RANGE,
                  #'ATR_mult': ATR_MULT_RANGE,
                  'ATR_PP_chg_mult': ATR_MULT_RANGE}  # the multiple of ATR used to determine if two pivot points are far enough apart}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        return input_params_perm

    ##################################################################################

    class df_col_headers:
        ATR_header = None

    class trades_list:
        current = []
        completed = []
        prev_avg_HA_OHLC = []
        first_new_high_low = None
        first_new_low_high = None

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""
        # candle_num
        data_df = data_df.reset_index()

        # heiken ashi
        data_df = technical_indicators.heiken_ashi(df=data_df)

        data_df['new_high'] = np.where(data_df['HA High'] > data_df['HA High'].shift(1),1,0)
        data_df['new_low'] = np.where(data_df['HA Low'] < data_df['HA Low'].shift(1),1,0)

        #data_df['prev_HA_red_green'] = data_df['HA_red_green'].shift(1)
        # data_df['is_pivot_point'] = np.where(data_df['prev_HA_red_green'] != data_df['HA_red_green'], 1,0)
        #
        # data_df['pivot_price'] = np.where(data_df['HA_red_green'] == 'green', data_df['HA Open'],data_df['HA Close'])


        data_df["consec_new_highs"] = data_df.groupby((data_df["new_high"] == 0).cumsum()).cumcount().shift(1)
        data_df["consec_new_lows"] = data_df.groupby((data_df["new_low"] == 0).cumsum()).cumcount().shift(1)

        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # get list of all indicators and drop rows of those columns with NA vals
        orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

        data_df = data_df.dropna(subset=indicator_col_names)
        data_df = data_df.reset_index(drop=True)
        return data_df

    def get_trades(frame):

        def record_trade(trade_obj):

            """
            convert class instance variables to dict, append dict to trades_list
            """

            trade_dict = {}

            for k, v in vars(trade_obj).items():
                if k.startswith('_') or k not in STATIC_TRADE_METRICS:
                    continue
                trade_dict[k] = v

            trades_list.completed.append(trade_dict)

        def exit_trade(frame,trade_obj):
            trade_obj.exit_price = frame['Close']
            trade_obj.exit_datetime = frame['Date_Time']
            return trade_obj

        def enter_trade(frame, bias):
            # create current trade object
            curr_trade = trade_metrics()

            curr_trade.trade_num = len(trades_list.completed) + len(trades_list.current) + 1
            curr_trade.bias = bias
            curr_trade.entry_datetime = frame['Date_Time']
            curr_trade.entry_price = frame['Close']

            if bias == 'buy':
                curr_trade.risk_price = curr_trade.entry_price - frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price + (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            elif bias == 'sell':
                curr_trade.risk_price = curr_trade.entry_price + frame[df_col_headers.ATR_header]
                curr_trade.profit_target = curr_trade.entry_price - (abs(curr_trade.entry_price - curr_trade.risk_price) * params['R_mult_PT'])

            else:
                print('PROBLEM ACQUIRING BIAS')
                exit()

            return curr_trade

        def entry_criteria_met(frame):

            def potential_entry(frame):

                if frame['consec_new_highs'] > params['consec_candles'] and frame['new_low'] == 1:
                    return True
                if frame['consec_new_lows'] > params['consec_candles'] and frame['new_high'] == 1:
                    return True

                return False

            def check_volatilty(frame):
                cutoff = frame[df_col_headers.ATR_header] * params['ATR_PP_chg_mult']
                pivot_difference = abs(frame['pivot_price'] - trades_list.prev_avg_HA_OHLC[1])
                #print('     ATR cutoff:',cutoff)
                #print(      trades_list.prev_avg_HA_OHLC)
                #print('     pivot diffs:',pivot_difference)

                if pivot_difference > cutoff:
                    #print('     pivot diff > cutoff...')
                    return True

                return False

            ################################################################################

            # if this is a pivot point and theres been x consecutive same color candles
            if potential_entry(frame=frame):
                #print(frame['Date_Time'])
                #print('is pivot point')

                if check_volatilty(frame=frame):
                    if frame['HA_red_green'] == 'red':
                        #print('             ENTERING SELL POSITION')
                        #print('-----------------------------------------------------------------------')
                        return 'sell'

                    if frame['HA_red_green'] == 'green':
                        #print('         ENTERING BUY POSITION')
                        #print('-----------------------------------------------------------------------')
                        return 'buy'

            #print('-----------------------------------------------------------------------')
            return False

        def exit_criteria_met(frame, trade_obj):

            if trade_obj.bias == 'buy':
                if frame['HA Close'] >= trade_obj.profit_target or frame['HA Close'] <= trade_obj.risk_price:
                    return True

            elif trade_obj.bias == 'sell':
                if frame['HA Close'] <= trade_obj.profit_target or frame['HA Close'] >= trade_obj.risk_price:
                    return True

            return False

        def handle_volatility_high_low(frame):
            if not trades_list.first_new_high_low and frame['new_high'] == 1:
                trades_list.first_new_high_low = frame['HA Low']

            elif trades_list.first_new_high_low and frame['new_high'] == 0:
                trades_list.first_new_high_low = None

            if not trades_list.first_new_low_high and frame['new_low'] == 1:
                trades_list.first_new_low_high = frame['HA High']

            elif trades_list.first_new_low_high and frame['new_low'] == 0:
                trades_list.first_new_low_high = None

            print(frame['Date_Time'])
            print(trades_list.first_new_high_low)
            print(trades_list.first_new_low_high)
            print('---------------------------------------------------')
            print('---------------------------------------------------')
            print('---------------------------------------------------')

            return

            ###################################################################

        handle_volatility_high_low(frame=frame)
        return
        # MONITOR ONGOING TRADES
        # do you have any ongoing trades? (yes)
        if trades_list.current:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through uncompleted trades list
            for trade in trades_list.current:
                # is exit criteria is met for that trade (yes)
                if exit_criteria_met(frame=frame,trade_obj=trade):
                    # exit and record the trade
                    trade = exit_trade(frame=frame,trade_obj=trade)
                    record_trade(trade_obj=trade)

                # if exit criteria not met for that trade then append to temp list
                else:
                    temp_list.append(trade)

            trades_list.current = temp_list

        # LOOK FOR NEW TRADES
        # do you have more than 3 trades on at once? (no)
        if len(trades_list.current) < 3:
            # has the entry criteria been met? (yes)
            trade_bias = entry_criteria_met(frame=frame)
            if trade_bias:
                # enter the trade
                curr_trade = enter_trade(frame=frame,bias=trade_bias)

                # append trade to uncompleted trades list
                trades_list.current.append(curr_trade)

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['HA Open'], high=data_df['HA High'],
                                     low=data_df['HA Low'], close=data_df['HA Close']), row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

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

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    data_df.apply(get_trades, axis=1)

    if do_chart:
        view_chart(trade_data=trades_list.completed)

    return trades_list.completed

def leg_std_dev_EXPERIMENTAL(params,data_df,do_chart=False):
    """
    use the rolling standard deviation of leg lengths for bias
    enter on... moving avg crossover idfk

    judge risk/base reward on ATR


    *** will need to calculate your legs on a per iteration basis
    """

    if params == 'test':
        params = {'ATR_num':ATR_NUM_RANGE,
                  #'ATR_mult':1,
                  'HL_window':6,
                  'leg_std_dev_win':14,
                  'leg_std_dev_cutoff':1.5}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        return input_params_perm

    ##################################################################################

    def get_swing_HL():

        def confirm_swing_point(HL_list, window_size):

            if len(HL_list) < window_size:
                return

            # get latest x points
            HL_list = HL_list[len(HL_list) - window_size:len(HL_list)]

            if all(x == HL_list[0] for x in HL_list):
                return HL_list[-1]
        #########################################################################

        prev_x_candles = data_df.iloc[idx - (params['HL_window'] - 1):idx + 1]

        idx_of_body_max = prev_x_candles['OC_max'].idxmax()
        idx_of_body_min = prev_x_candles['OC_min'].idxmin()
        idx_of_wick_max = prev_x_candles['High'].idxmax()
        idx_of_wick_min = prev_x_candles['Low'].idxmin()

        temp_body_high_list.append([prev_x_candles.loc[idx_of_body_max]['Date_Time'], prev_x_candles.loc[idx_of_body_max]['OC_max'],'body high'])
        temp_body_low_list.append([prev_x_candles.loc[idx_of_body_min]['Date_Time'], prev_x_candles.loc[idx_of_body_min]['OC_min'],'body low'])
        temp_wick_high_list.append([prev_x_candles.loc[idx_of_wick_max]['Date_Time'], prev_x_candles.loc[idx_of_wick_max]['High'],'wick high'])
        temp_wick_low_list.append([prev_x_candles.loc[idx_of_wick_min]['Date_Time'], prev_x_candles.loc[idx_of_wick_min]['Low'],'wick low'])

        #body HL
        new_SP = confirm_swing_point(HL_list=temp_body_high_list, window_size=params['HL_window'])
        if new_SP:
            swing_points['body'].append(new_SP)

        new_SP = confirm_swing_point(HL_list=temp_body_low_list, window_size=params['HL_window'])
        if new_SP:
            swing_points['body'].append(new_SP)

        #wick HL
        new_SP = confirm_swing_point(HL_list=temp_wick_high_list,window_size=params['HL_window'])
        if new_SP:
            swing_points['wick'].append(new_SP)

        new_SP = confirm_swing_point(HL_list=temp_wick_high_list, window_size=params['HL_window'])
        if new_SP:
            swing_points['wick'].append(new_SP)

    def get_leg_lengths(body_wick,prev_len):
        """

        """

        def Sort(sort_list):

            # reverse = None (Sorts in Ascending order)
            # key is set to sort using second element of
            # sublist lambda has been used
            sort_list.sort(key=lambda x: x[0])
            return sort_list

        def validate_legs(sorted_list):
            # for highs to low, are there highs higher than past highs and vice versa, dont want those
            ret_list = []

            for i in range(0, len(sorted_list)):
                curr_1 = sorted_list[i + 1]
                prev_1 = sorted_list[i]

                if curr_1[2] == 'body high':
                    if curr_1[1] < prev_1[1]:
                        if prev_1 not in ret_list:
                            ret_list.append(prev_1)

                    if curr_1 not in ret_list:
                        ret_list.append(curr_1)
                if i == (len(sorted_list)-1):
                    break


            return ret_list

        def look_for_same_points():
            ret_list = []
            for i in range(len(swing_points['body']) - 2, 0, -1):

                curr_1 = swing_points['body'][i]
                prev_1 = swing_points['body'][i - 1]

                # if the curr and prev are equal append them both to same points
                if curr_1[2] == prev_1[2]:
                    if curr_1 not in same_points:
                        ret_list.append(curr_1)
                    if prev_1 not in same_points:
                        ret_list.append(prev_1)
                else:  # otherwise stop the loop
                    break

            return ret_list

        ##################################################################################

        # has the list changed?
        if len(swing_points[body_wick] > prev_len):
            curr = swing_points[body_wick][-1]
            prev = swing_points[body_wick][-2]

            same_points = []

            # compare the last two points
            if curr[2] != prev[2]:
                # if points arent equal
                # look backward through swing points list starting with prev point to look for same consecutive points i.e [H,H,H,L]
                same_points = look_for_same_points()

                # if same points list is occupied
                if same_points:

                    # sort samepoints list by chronological order from oldest to latest
                    same_points = Sort(sort_list=same_points)

                    # make sure the legs make sense
                    same_points = validate_legs(sorted_list=same_points)

                    # loop through same points, get distance from point to current opposite point, append
                    for point in same_points:
                        leg_lengths[body_wick].append([point[0], curr[0], abs(point[1] - curr[1])])

                else: #otherwise get distance from current to previous and append
                    leg_lengths[body_wick].append([prev[0], curr[0], abs(curr[1] - prev[1])])

    def get_legs_std_dev():

        body = []
        wick = []

        for i in swing_points['body']:
            body.append(i[1])

        for i in swing_points['wick']:
            wick.append(i[1])

        body_leg_std_dev = statistics.pstdev(body)
        wick_leg_std_dev = statistics.pstdev(wick)

    ####################################################################################################################

    # records highs and lows for each iteration of the rolling window
    temp_body_high_list = []
    temp_body_low_list = []
    temp_wick_high_list = []
    temp_wick_low_list = []

    # to track if changes have occured in swing point lists
    prev_body_len = 0
    prev_wick_len = 0

    body_leg_std_dev = None
    wick_leg_std_dev = None

    swing_points = {'body':[],
                    'wick':[]}

    leg_lengths = {'body':[],
                   'wick':[]}

    data_df['OC_max'] = data_df[["Open", "Close"]].max(axis=1)
    data_df['OC_min'] = data_df[["Open", "Close"]].min(axis=1)

    for idx in data_df.index:
        print(idx)
        if idx < params['HL_window']:
            continue

        curr_candle = data_df.loc[idx]

        # updates any new swing points
        get_swing_HL()

        # calculate legs and get price lengths
        get_leg_lengths(body_wick='body',prev_len=prev_body_len)
        get_leg_lengths(body_wick='wick',prev_len=prev_wick_len)

        # update swing point list lengths for next iteration
        prev_body_len = len(swing_points['body'])
        prev_wick_len = len(swing_points['wick'])

        get_legs_std_dev()

        """
        probably did all this for nothing, could have just found the swing points in the dataframe using what i already 
        set up then offset it by the relevant number of periods, loop through each candle and the swing points will get 
        confirmed as it would organically, then its just a matter of keeping track of a list of confirmed swing points,
        making a std dev calculation, making a bias calculation using the std dev multiple relative to price, 
        generating some sort of entry signal (or just use the std dev multiple to start) then recording the results. 
        You should probably do this since what you have here is going to take forever to run 10k iterations of
        """

    return

def basic_leg_std_dev(params,data_df,do_chart=False):
    """
    use the rolling standard deviation of leg lengths for buy sell entrys
    enter on close countering direction of leg once it surpases std dev cutoff

    profit exit on

    base risk/judge reward based on ATR
    
    trades will overlap
    """

    if params == 'test':
        params = {'ATR_num':ATR_NUM_RANGE,
                  #'ATR_mult':1,
                  'R_mult_PT':R_PROF_TARG_RANGE,
                  'HL_window':6,
                  'leg_std_dev_win':14,
                  'leg_std_dev_cutoff':1.5,
                  'use_body':BINARY_RANGE} #0 = use wick}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        return input_params_perm

    ##################################################################################

    class df_col_headers:
        ATR_header = None

    def update_swing_point_list():
        if curr_candle[high_str]:
            swing_point_list.append([curr_candle['Date_Time'], curr_candle[high_str], high_str])
        if curr_candle[low_str]:
            swing_point_list.append([curr_candle['Date_Time'], curr_candle[low_str], low_str])

    def find_new_legs():

        def sort_list(input_list):
            input_list.sort(key=lambda x: x[0])
            return input_list

        def validate_legs(sorted_list):

            def do_append(item):
                if item not in ret_list:
                    ret_list.append(item)

            ######################################################

            # for highs to low, are there highs higher than past highs and vice versa, dont want those
            ret_list = []

            for i in range(0, len(sorted_list)):
                if (i+1) > (len(sorted_list)-1):
                    break

                curr_1 = sorted_list[i + 1]
                prev_1 = sorted_list[i]


                # is the current point a high
                if curr_1[2] == high_str:
                    # if curr price is greater than prev price
                    if curr_1[1] < prev_1[1]:
                        # append prev point
                        do_append(item=prev_1)

                    # then append curr point
                    do_append(item=curr_1)

                # is the current point a low
                if curr_1[2] == low_str:
                    # if curr price is less than prev price
                    if curr_1[1] > prev_1[1]:
                        # append prev point
                        do_append(item=prev_1)
                    # then append curr point
                    do_append(item=curr_1)

                if i == (len(sorted_list) - 1):
                    break

            return ret_list

        def look_for_same_points():
            ret_list = []
            for i in range(len(swing_point_list) - 2, 0, -1):

                curr_1 = swing_point_list[i]
                prev_1 = swing_point_list[i - 1]

                # if the curr and prev are equal append them both to same points
                if curr_1[2] == prev_1[2]:
                    if curr_1 not in same_points:
                        ret_list.append(curr_1)
                    if prev_1 not in same_points:
                        ret_list.append(prev_1)
                else:  # otherwise stop the loop
                    break

            return ret_list

        #########################################################

        # has the swing point list changed?
        if len(swing_point_list) > prev_swing_point_list_len and len(swing_point_list) > 1:

            curr = swing_point_list[-1]
            prev = swing_point_list[-2]

            same_points = []

            # compare the last two points
            if curr[2] != prev[2]:
                # if points arent equal
                # look backward through swing points list starting with prev point to look for same consecutive points i.e [H,H,H,L]
                same_points = look_for_same_points()

                # if same points list is occupied
                if same_points:

                    # sort samepoints list by chronological order from oldest to latest
                    same_points = sort_list(input_list=same_points)

                    # make sure the legs make sense
                    same_points = validate_legs(sorted_list=same_points)

                    # loop through same points, get distance from point to current opposite point, append
                    for point in same_points:

                        leg_list.append([[point[0],point[1]], [curr[0],curr[1]], abs(point[1] - curr[1])])

                else:  # otherwise get distance from current to previous and append
                    leg_list.append([[prev[0],prev[1]], [curr[0],curr[1]], abs(curr[1] - prev[1])])

    def update_bias():

        # get the most current ongoing leg direction
        if (swing_point_list[-1][1] - curr_candle['Close']) > 0:
            leg_direction = 'Down'
        elif (swing_point_list[-1][1] - curr_candle['Close']) < 0:
            leg_direction = 'Up'

        # get the most current ongoing leg length
        curr_leg_len = abs(swing_point_list[-1][1] - curr_candle['Close'])

        cutoff = abs(leg_std_dev * params['leg_std_dev_cutoff'])
        # is the curr leg length > std deviation * leg std dev cutoff and is going down
        if curr_leg_len > cutoff and leg_direction == 'Down':
            return 'buy'

        # if the curr leg length > std deviation * std dev cutoff and is going up
        elif curr_leg_len > abs(leg_std_dev * params['leg_std_dev_cutoff']) and leg_direction == 'Up':
            return 'sell'

        return curr_bias

    def calculate_leg_std_dev():

        # is there a change in the list or if the the list is full and the leg std dev hasnt been filled yet
        if (len(leg_list) > prev_leg_list_len) or (len(leg_list) >= prev_leg_list_len and not leg_std_dev):

            leg_difs = []

            for leg in leg_list:
                leg_difs.append(leg[2])

            return statistics.pstdev(leg_difs[-params['leg_std_dev_win']:])

        return leg_std_dev

    def do_indicator_calculations(df):
        df = technical_indicators.average_true_range(df=df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        df = technical_indicators.swing_HL(df=df, window_size=params['HL_window'])

        # offset the swing point columns into the future (HL points arent confirmed until HL window periods pass)
        df[['body_swing_high', 'body_swing_low', 'wick_swing_high', 'wick_swing_low']] = df[['body_swing_high', 'body_swing_low', 'wick_swing_high', 'wick_swing_low']].shift(params['HL_window'])

        df['body_swing_high'] = df['body_swing_high'].fillna(False)
        df['body_swing_low'] = df['body_swing_low'].fillna(False)
        df['wick_swing_high'] = df['wick_swing_high'].fillna(False)
        df['wick_swing_low'] = df['wick_swing_low'].fillna(False)

        return df

    def entry_criteria_met():

        if curr_bias != prev_bias:
            return True

    def enter_trade(trade_num):

        # compare the close to the last swing point price
        if curr_candle['Close'] > swing_point_list[-1][1]:
            curr_trade.bias = 'sell'
        elif curr_candle['Close'] < swing_point_list[-1][1]:
            curr_trade.bias = 'buy'

        curr_trade.trade_num = trade_num
        curr_trade.entry_datetime = curr_candle['Date_Time']
        curr_trade.entry_price = curr_candle['Close']

        if curr_trade.bias == 'buy':
            curr_trade.risk_price = curr_trade.entry_price - curr_candle[df_col_headers.ATR_header]
        else:
            curr_trade.risk_price = curr_trade.entry_price + curr_candle[df_col_headers.ATR_header]

    def exit_criteria_met(trade_obj):

        def loss_exit_hit():  # is ATR risk price
            if trade_obj.bias == 'buy':
                if curr_candle['Close'] < trade_obj.risk_price:
                    return True
            if trade_obj.bias == 'sell':
                if curr_candle['Close'] > trade_obj.risk_price:
                    return True

            return False

        def profit_exit_hit():
            if trade_obj.bias == 'buy':
                PT = trade_obj.entry_price + (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] > PT:
                    return True

            if trade_obj.bias == 'sell':
                PT = trade_obj.entry_price - (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] < PT:
                    return True

        ########################################

        if loss_exit_hit() or profit_exit_hit():
            return True

    def exit_trade(trade_obj):
        trade_obj.exit_price = curr_candle['Close']
        trade_obj.exit_datetime = curr_candle['Date_Time']
        return trade_obj

    def record_trade(trade_obj):
        """
        convert class instance variables to dict, append dict to trades_list
        """

        trade_dict = {}

        for k, v in vars(trade_obj).items():
            if k.startswith('_'):
                continue
            trade_dict[k] = v

        completed_trades_list.append(trade_dict)

        # if do_chart:
        #     view_chart(trade_data=completed_trades_list, leg_list=leg_list)
        #     print()

    def view_chart(trade_data, leg_list):
        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)

        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

        for leg in leg_list:
            x0 = leg[0][0]
            x1 = leg[1][0]
            y0 = leg[0][1]
            y1 = leg[1][1]

            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='black', size=6)),row=1, col=1)

        for point in swing_point_list:
            #[Timestamp('2019-01-10 20:00:00'), 1.15608, 'body_swing_high']
            x0 = point[0]
            y0 = point[1]

            if 'high' in point[2]:
                color = 'blue'
            else:
                color = 'teal'

            fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers', marker=dict(color=color, size=6)), row=1,col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)

        fig.show()

    ####################################################################################################################

    if params['use_body'] == 1:
        high_str = 'body_swing_high'
        low_str = 'body_swing_low'
    elif params['use_body'] == 0:
        high_str = 'wick_swing_high'
        low_str = 'wick_swing_low'

    swing_point_list = [] # date, price, is high/low
    leg_list = [] #[start date, price],[end date, price], length
    prev_swing_point_list_len = 0
    prev_leg_list_len = 0

    leg_std_dev = None

    data_df = do_indicator_calculations(df=data_df)

    curr_trade = trade_metrics()
    uncompleted_trades_list = []
    completed_trades_list = []

    curr_bias = None
    prev_bias = None

    trade_num = 0
    for idx in data_df.index:
        curr_candle = data_df.loc[idx]

        # update swing points, find new legs, calculate the std dev
        update_swing_point_list()
        find_new_legs()

        prev_swing_point_list_len = len(swing_point_list)
        prev_leg_list_len = len(leg_list)

        # if the strategy has enough data to start running
        if len(leg_list) >= params['leg_std_dev_win']:

            leg_std_dev = calculate_leg_std_dev()
            curr_bias = update_bias()

            # are you currently in any trades
            if uncompleted_trades_list:
                # creat a blank temp list to store trades
                temp_list = []

                for trade in uncompleted_trades_list:
                    if exit_criteria_met(trade_obj=trade):
                        trade = exit_trade(trade_obj=trade)
                        record_trade(trade_obj=trade)

                    else:
                        temp_list.append(trade)

                uncompleted_trades_list = temp_list

            # are there any new trades
            if len(uncompleted_trades_list) < 3:

                if entry_criteria_met():
                    enter_trade(trade_num)
                    uncompleted_trades_list.append(curr_trade)
                    curr_trade = trade_metrics()
                    trade_num += 1

            prev_bias = curr_bias

    if do_chart:
        view_chart(trade_data=completed_trades_list, leg_list=leg_list)

    return completed_trades_list


#!!! keeping for reference
def basic_ichimoku_V1(params,data_df,do_chart=False):
    """
    enter long when
        - price closes above projected cloud
        - current cloud is green
        - conversion line is above base line
        - lagging line above lagging line projected cloud
    enter short on opposite

    does use the lagging line (chikou)

    based risk/judge reward on atr

    trades may overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """

    strat_params = {'ATR_num':ATR_NUM_RANGE,
                  #'ATR_mult':ATR_MULT_RANGE,
                  'ATR_PP_chg_mult':ATR_MULT_RANGE, # the multiple of ATR used to determine if two pivot points are far enough apart
                  'R_mult_PT':R_PROF_TARG_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        return input_params_perm

    ##################################################################################

    class df_col_headers:
        ATR_header = None

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""

        def get_price_to_cloud_bias(dataframe):
            if dataframe['Close'] > dataframe['projected_cloud_max']:
                return 'long'
            elif dataframe['Close'] < dataframe['projected_cloud_min']:
                return 'short'
            else:
                return 'None'

        def get_curr_cloud_bias(dataframe):
            if dataframe["Current Senkou A"] > dataframe["Current Senkou B"]:
                return 'long'
            elif dataframe["Current Senkou A"] < dataframe["Current Senkou B"]:
                return 'short'
            else:
                return 'None'

        def get_conversion_to_base_bias(dataframe):
            if dataframe['Tenkan'] > dataframe['Kijun']:
                return 'long'
            elif dataframe['Tenkan'] <= dataframe['Kijun']:
                return 'short'

        def get_chikou_bias(dataframe):
            if dataframe['Chikou'] > dataframe['chikou_cloud_max']:
                return 'long'
            elif dataframe['Chikou'] < dataframe['chikou_cloud_min']:
                return 'short'
            else:
                return 'None'

        def get_buy_sell_bias(dataframe):
            if dataframe['bias_1'] == 'long' and dataframe['bias_2'] == 'long' and dataframe['bias_3'] == 'long' and dataframe['bias_4'] == 'long':
                return 'buy'
            elif dataframe['bias_1'] == 'short' and dataframe['bias_2'] == 'short' and dataframe['bias_3'] == 'short' and dataframe['bias_4'] == 'short':
                return 'sell'
            else:
                return 'no_bias'

        ####################################################################################
        # candle_num
        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # ichimoku
        data_df = technical_indicators.ichimoku(df=data_df)

        data_df['projected_cloud_max'] = data_df[["Projected Senkou A", "Projected Senkou B"]].max(axis=1)
        data_df['projected_cloud_min'] = data_df[["Projected Senkou A", "Projected Senkou B"]].min(axis=1)

        data_df['chikou_cloud_max'] = data_df[['Chikou adj Senkou B', 'Chikou adj Senkou A']].max(axis=1)
        data_df['chikou_cloud_min'] = data_df[['Chikou adj Senkou B', 'Chikou adj Senkou A']].min(axis=1)

        data_df['bias_1'] = data_df.apply(get_price_to_cloud_bias, axis=1)
        data_df['bias_2'] = data_df.apply(get_curr_cloud_bias, axis=1)
        data_df['bias_3'] = data_df.apply(get_conversion_to_base_bias, axis=1)
        data_df['bias_4'] = data_df.apply(get_chikou_bias, axis=1)

        data_df['buy_sell'] = data_df.apply(get_buy_sell_bias, axis=1)

        data_df = data_df.dropna()

        data_df = data_df.reset_index()

        return data_df

    def entry_criteria_met():
        if curr_bias != prev_bias:
            if curr_bias == 'buy' or curr_bias == 'sell':
                return True

    def enter_trade(trade_num):
        curr_trade.bias = curr_bias
        curr_trade.trade_num = trade_num
        curr_trade.entry_datetime = curr_candle['Date_Time']
        curr_trade.entry_price = curr_candle['Close'] # entering on the break of the previous candle high

        if curr_trade.bias == 'buy':
            curr_trade.risk_price = curr_trade.entry_price - curr_candle[df_col_headers.ATR_header]
        if curr_trade.bias == 'sell':
            curr_trade.risk_price = curr_trade.entry_price + curr_candle[df_col_headers.ATR_header]

    def exit_trade(trade_obj):
        if trade_obj.bias == 'buy':
            trade_obj.exit_price = curr_candle['Close']
        if trade_obj.bias == 'sell':
            trade_obj.exit_price = curr_candle['Close']

        trade_obj.exit_datetime = curr_candle['Date_Time']

    def exit_criteria_met(trade_obj):

        def loss_exit_hit():  # is ATR risk price
            if trade_obj.bias == 'buy':
                if curr_candle['Close'] < trade_obj.risk_price:
                    return True
            if trade_obj.bias == 'sell':
                if curr_candle['Close'] > trade_obj.risk_price:
                    return True

            return False

        def target_exit_hit():
            if trade_obj.bias == 'buy':
                PT = trade_obj.entry_price + (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] > PT:
                    return True

            if trade_obj.bias == 'sell':
                PT = trade_obj.entry_price - (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] < PT:
                    return True

            return False

        ########################################

        if loss_exit_hit() or target_exit_hit():
            return True

    def record_trade(trade_obj):

        """
        convert class instance variables to dict, append dict to trades_list
        """

        trade_dict = {}

        # get all original variable names
        accepted_params_list = [attr for attr in dir(trade_metrics()) if not callable(getattr(trade_metrics(), attr)) and not attr.startswith("__")]

        for k, v in vars(trade_obj).items():
            if k.startswith('_'):
                continue
            if k not in accepted_params_list:
                continue
            trade_dict[k] = v

        completed_trades_list.append(trade_dict)

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # conversion line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Tenkan'], mode='lines', name='conversion line',
                       line=dict(color='orange')),row=1, col=1)

        # base line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Kijun'], mode='lines', name='base line',
                       line=dict(color='blue')),row=1, col=1)

        # leading span A
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Projected Senkou A'], mode='lines', name='leading span A',
                       line=dict(color='green')),row=1, col=1)

        # leading span B
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Projected Senkou B'], mode='lines', name='leading span B',
                       line=dict(color='red')),row=1, col=1)

        # chikou
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Chikou'], mode='lines', name='Chikou',
                       line=dict(color='purple')), row=1, col=1)


        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)

        fig.show()

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    # call the class instance
    curr_trade = trade_metrics()

    uncompleted_trades_list = []
    completed_trades_list = []

    curr_bias = None
    prev_bias = None

    trade_num = 0
    # loop through df
    for idx in data_df.index:

        # get the current candle data
        curr_candle = data_df.loc[idx]

        curr_bias = curr_candle['buy_sell']

        # are you in trades
        if uncompleted_trades_list:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through trades
            for trade in uncompleted_trades_list:
                # test for exit criteria
                if exit_criteria_met(trade_obj=trade):
                    exit_trade(trade_obj=trade)
                    record_trade(trade_obj=trade)

                else:
                    temp_list.append(trade)

            uncompleted_trades_list = temp_list

        # look for more trade entries
        # is your entry criteria met?
        if len(uncompleted_trades_list) < 3:
            # if the entry criteria is met...
            if entry_criteria_met():
                enter_trade(trade_num)  # enter the trade
                trade_num += 1  # update the trade num
                uncompleted_trades_list.append(curr_trade)  # append the curr trade class obj to uncomp trades list
                curr_trade = trade_metrics()  # reset the curr trade class obj

        prev_bias = curr_bias

    if do_chart:
        view_chart(trade_data=completed_trades_list)

    return completed_trades_list

# candle_data = pd.read_csv('Historical_Data/Forex/EUR_USD_4_hour.csv')
# candle_data['Date_Time'] = pd.to_datetime(candle_data['Date_Time'])
#basic_ichimoku_V1_ABRIDGED(params='test',data_df=candle_data,do_chart=True)

def basic_ichimoku_V2(params,data_df,do_chart=False):
    """
    enter long when
        - price closes above projected cloud
        - current cloud is green
        - conversion line is above base line
        - lagging line above lagging line projected cloud
    enter short on opposite

    does NOT use the lagging line (chikou)

    based risk/judge reward on atr

    trades may overlap (limit of 3 trades on at once)
        - keeps from getting over exposed during choppy consolidation times
    """
    strat_params = {'ATR_num':ATR_NUM_RANGE,
                  #'ATR_mult':ATR_MULT_RANGE,
                  'ATR_PP_chg_mult':ATR_MULT_RANGE, # the multiple of ATR used to determine if two pivot points are far enough apart
                  'R_mult_PT':R_PROF_TARG_RANGE}

    ################################################################################

    if params == 'test':
        # take the first item in each param list

        ret_params = {k: v[0] for (k, v) in strat_params.items()}

        ret_params = {}
        for k, v in strat_params.items():
            ret_params[k] = v[0]
        params = ret_params

    elif params == 'get perms':
        input_params_perm = create_permuations(params_dict=strat_params)
        return input_params_perm

    ##################################################################################

    class df_col_headers:
        ATR_header = None

    def do_indicator_calculations(data_df):
        """calculate indicators and chop data down to the slowest one"""

        def get_price_to_cloud_bias(dataframe):
            if dataframe['Close'] > dataframe['projected_cloud_max']:
                return 'long'
            elif dataframe['Close'] < dataframe['projected_cloud_min']:
                return 'short'
            else:
                return 'None'

        def get_curr_cloud_bias(dataframe):
            if dataframe["Current Senkou A"] > dataframe["Current Senkou B"]:
                return 'long'
            elif dataframe["Current Senkou A"] < dataframe["Current Senkou B"]:
                return 'short'
            else:
                return 'None'

        def get_conversion_to_base_bias(dataframe):
            if dataframe['Tenkan'] > dataframe['Kijun']:
                return 'long'
            elif dataframe['Tenkan'] <= dataframe['Kijun']:
                return 'short'

        def get_buy_sell_bias(dataframe):
            if dataframe['bias_1'] == 'long' and dataframe['bias_2'] == 'long' and dataframe['bias_3'] == 'long':
                return 'buy'
            elif dataframe['bias_1'] == 'short' and dataframe['bias_2'] == 'short' and dataframe['bias_3'] == 'short':
                return 'sell'
            else:
                return 'no_bias'

        ####################################################################################
        # candle_num
        # ATR
        data_df = technical_indicators.average_true_range(df=data_df, window=params['ATR_num'])
        df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

        # ichimoku
        data_df = technical_indicators.ichimoku(df=data_df)

        data_df['projected_cloud_max'] = data_df[["Projected Senkou A", "Projected Senkou B"]].max(axis=1)
        data_df['projected_cloud_min'] = data_df[["Projected Senkou A", "Projected Senkou B"]].min(axis=1)

        data_df['chikou_cloud_max'] = data_df[['Chikou adj Senkou B', 'Chikou adj Senkou A']].max(axis=1)
        data_df['chikou_cloud_min'] = data_df[['Chikou adj Senkou B', 'Chikou adj Senkou A']].min(axis=1)

        data_df['bias_1'] = data_df.apply(get_price_to_cloud_bias, axis=1)
        data_df['bias_2'] = data_df.apply(get_curr_cloud_bias, axis=1)
        data_df['bias_3'] = data_df.apply(get_conversion_to_base_bias, axis=1)

        data_df['buy_sell'] = data_df.apply(get_buy_sell_bias, axis=1)

        data_df = data_df.dropna()

        data_df = data_df.reset_index()

        return data_df

    def entry_criteria_met():
        if curr_bias != prev_bias:
            if curr_bias == 'buy' or curr_bias == 'sell':
                return True

    def enter_trade(trade_num):
        curr_trade.bias = curr_bias
        curr_trade.trade_num = trade_num
        curr_trade.entry_datetime = curr_candle['Date_Time']
        curr_trade.entry_price = curr_candle['Close'] # entering on the break of the previous candle high

        if curr_trade.bias == 'buy':
            curr_trade.risk_price = curr_trade.entry_price - curr_candle[df_col_headers.ATR_header]
        if curr_trade.bias == 'sell':
            curr_trade.risk_price = curr_trade.entry_price + curr_candle[df_col_headers.ATR_header]

    def exit_trade(trade_obj):
        if trade_obj.bias == 'buy':
            trade_obj.exit_price = curr_candle['Close']
        if trade_obj.bias == 'sell':
            trade_obj.exit_price = curr_candle['Close']

        trade_obj.exit_datetime = curr_candle['Date_Time']

    def exit_criteria_met(trade_obj):

        def loss_exit_hit():  # is ATR risk price
            if trade_obj.bias == 'buy':
                if curr_candle['Close'] < trade_obj.risk_price:
                    return True
            if trade_obj.bias == 'sell':
                if curr_candle['Close'] > trade_obj.risk_price:
                    return True

            return False

        def target_exit_hit():
            if trade_obj.bias == 'buy':
                PT = trade_obj.entry_price + (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] > PT:
                    return True

            if trade_obj.bias == 'sell':
                PT = trade_obj.entry_price - (abs(trade_obj.risk_price - trade_obj.entry_price) * params['R_mult_PT'])
                if curr_candle['Close'] < PT:
                    return True

            return False

        ########################################

        if loss_exit_hit() or target_exit_hit():
            return True

    def record_trade(trade_obj):

        """
        convert class instance variables to dict, append dict to trades_list
        """

        trade_dict = {}

        # get all original variable names
        accepted_params_list = [attr for attr in dir(trade_metrics()) if not callable(getattr(trade_metrics(), attr)) and not attr.startswith("__")]

        for k, v in vars(trade_obj).items():
            if k.startswith('_'):
                continue
            if k not in accepted_params_list:
                continue

            trade_dict[k] = v

        completed_trades_list.append(trade_dict)

    def view_chart(trade_data):

        """
        :param data_df: the input data as dataframe
        :param trade_data: the trade data as a list of dicts
        """

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Candlestick(x=data_df['Date_Time'], open=data_df['Open'], high=data_df['High'],
                                     low=data_df['Low'], close=data_df['Close']), row=1, col=1)
        # conversion line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Tenkan'], mode='lines', name='conversion line',
                       line=dict(color='orange')),row=1, col=1)

        # base line
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Kijun'], mode='lines', name='base line',
                       line=dict(color='blue')),row=1, col=1)

        # leading span A
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Projected Senkou A'], mode='lines', name='leading span A',
                       line=dict(color='green')),row=1, col=1)

        # leading span B
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Projected Senkou B'], mode='lines', name='leading span B',
                       line=dict(color='red')),row=1, col=1)

        # chikou
        fig.add_trace(
            go.Scatter(x=data_df['Date_Time'], y=data_df['Chikou'], mode='lines', name='Chikou',
                       line=dict(color='purple')), row=1, col=1)


        fig.add_trace(go.Bar(x=data_df['Date_Time'], y=data_df['Volume']), row=2, col=1, secondary_y=False)

        for trade in trade_data:
            # mark start candle
            x0 = x1 = trade['entry_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()
            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='green', size=9),
                           name=str(trade['trade_num']) + ' ' + trade['bias'] + ' entry'),
                row=1, col=1)

            # mark entry candle
            x0 = x1 = trade['exit_datetime']
            y0 = data_df['High'].max()
            y1 = data_df['Low'].min()

            fig.add_trace(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', marker=dict(color='red', size=9),
                           name=str(trade['trade_num']) + ' exit'),
                row=1, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)

        fig.show()

    ###########################################################

    data_df = do_indicator_calculations(data_df=data_df)

    # call the class instance
    curr_trade = trade_metrics()

    uncompleted_trades_list = []
    completed_trades_list = []

    curr_bias = None
    prev_bias = None

    trade_num = 0
    # loop through df
    for idx in data_df.index:

        # get the current candle data
        curr_candle = data_df.loc[idx]

        curr_bias = curr_candle['buy_sell']

        # are you in trades
        if uncompleted_trades_list:
            # creat a blank temp list to store trades
            temp_list = []

            # loop through trades
            for trade in uncompleted_trades_list:
                # test for exit criteria
                if exit_criteria_met(trade_obj=trade):
                    exit_trade(trade_obj=trade)
                    record_trade(trade_obj=trade)

                else:
                    temp_list.append(trade)

            uncompleted_trades_list = temp_list

        # look for more trade entries
        # is your entry criteria met?
        if len(uncompleted_trades_list) < 3:
            # if the entry criteria is met...
            if entry_criteria_met():
                enter_trade(trade_num)  # enter the trade
                trade_num += 1  # update the trade num
                uncompleted_trades_list.append(curr_trade)  # append the curr trade class obj to uncomp trades list
                curr_trade = trade_metrics()  # reset the curr trade class obj

        prev_bias = curr_bias

    if do_chart:
        view_chart(trade_data=completed_trades_list)

    return completed_trades_list



