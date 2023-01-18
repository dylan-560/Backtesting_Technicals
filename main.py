from datetime import datetime, timedelta
import json
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import time

import helper_functions
import mysql_connect
from settings import STRATEGIES_DICT, \
    MARKETS_BREAKDOWN, \
    TIMEFRAMES, \
    TOTAL_MONTHS_OF_DATA, \
    BACKTEST_MONTHS

class identifier_metrics():
    strat_id = None
    asset_id = None
    asset = None
    timeframe = None
    strategy = None
    strategy_params = None
    date_added = None

class results_metrics():
    period_end_date = None
    period_start_date = None
    number_of_bars = None
    trades_list = None
    total_realized_R = None
    strike_rate = None
    num_trades = None
    avg_hold_time = None
    expectancy = None
    avg_winner = None
    avg_loser = None
    largest_winner = None
    largest_loser = None
    max_drawdown = None
    max_drawup = None
    winners_std_dev_MAE = None
    winners_std_dev_MFE = None
    losers_std_dev_MAE = None
    losers_std_dev_MFE = None
    kelly_criterion = None
    returns_to_asset_corr = None
    equity_curve_regression_slope = None
    equity_curve_regression_std_error = None

def run_strategy(strat_id_metrics, candle_data, existing_record_dates, strategy_func): #TODO reconfigure this to make less queries, pull all the data up front and load into local memory

    def does_not_need_update(x,y):
        # compares existing record dates to required window dates
        # gets difference between two list of dicts, if no difference return True
        # True = does not need update
        test_list = [i for i in x + y if i not in x or i not in y]
        if len(test_list) == 0:
            return True
        return False

    ###################################################################################

    # dates start and end dates on record for a strat perm with particular params, asset id, strat id, and timeframe
    # existing_record_dates = mysql_connect.get_existing_eval_entry_dates(perm_params=strat_id_metrics.strategy_params,
    #                                                                     asset_id=strat_id_metrics.asset_id,
    #                                                                     strat_id=strat_id_metrics.strat_id,
    #                                                                     timeframe=strat_id_metrics.timeframe)

    # window dates for the current month start going back to BACKTEST_MONTHS
    required_window_dates = helper_functions.get_backtest_windows_date_list()

    #compare existing record dates to required window dates to see if database entry need to be updated?
    if does_not_need_update(x=existing_record_dates,y=required_window_dates):
        #print(strat_id_metrics.strategy_params,'DOES NOT NEED UPDATE')
        return

    # window dates needed for the update
    needed_windows = [d for d in required_window_dates if d not in existing_record_dates]

    # gets total date range of needed windows
    needed_windows_total_date_range = {}
    for window in needed_windows:
        needed_windows_total_date_range['range_end'] = max([e['month_end'] for e in needed_windows])
        needed_windows_total_date_range['range_start'] = min([s['month_start'] for s in needed_windows])

    # window dates outside of the required window date range
    not_in_required_range = [d for d in existing_record_dates if d not in required_window_dates]

    offset_candles = max([item for item in strat_id_metrics.strategy_params.values()])

    # use those months wittle down the inputed candle data
    backtest_df = helper_functions.slice_backtest_ohlcv_df(df=candle_data,
                                                           offset_candles=offset_candles,
                                                           start_date=needed_windows_total_date_range['range_start'],
                                                           end_date=needed_windows_total_date_range['range_end'])

    # make sure candle data is ordered correctly
    backtest_df = helper_functions.reorder_df(df=backtest_df,top_line_most_recent=False)


    # run the strategy and return the list of trades
    trades_list = strategy_func(params=strat_id_metrics.strategy_params,
                                data_df=backtest_df)

    # if there are no trades found
    if not trades_list:
        return

    strat_id_metrics.strategy_params = json.dumps(strat_id_metrics.strategy_params)

    # ret_entries = [] # TODO create instructions for return entries as to if they should update an existing entry or create a new entry

    # get the results for each window
    for window in needed_windows:
        # window = {'month_end': datetime.datetime(2022, 4, 1, 0, 0), 'month_start': datetime.datetime(2022, 3, 1, 0, 0)}
        window_strat_results_metrics = results_metrics()
        window_strat_results_metrics.period_end_date = window['month_end']
        window_strat_results_metrics.period_start_date = window['month_start']

        # only get trades from last whole month
        window_strat_results_metrics.trades_list = helper_functions.date_filter_trades(trades=trades_list,
                                                                                       start_date=window['month_start'],
                                                                                       end_date=window['month_end'])
        # if there are no trades found
        if not window_strat_results_metrics.trades_list:
            continue

        # get evaluation results
        window_strat_results_metrics = helper_functions.get_evaluation_metrics(class_obj=window_strat_results_metrics,
                                                                               trade_data=window_strat_results_metrics.trades_list,
                                                                               candle_data=candle_data)

        window_strat_results_metrics.trades_list = helper_functions.create_json_string(window_strat_results_metrics.trades_list)
        #strat_results_metrics.trades_list = json.dumps(strat_results_metrics.trades_list)

        window_entry_dict = {}
        for k,v in vars(strat_id_metrics).items():
            if not k.startswith('__'):
                window_entry_dict[k] = v

        for k,v in vars(window_strat_results_metrics).items():
            if not k.startswith('__'):
                if isinstance(v,float):
                    v = round(v,3)
                window_entry_dict[k] = v

        # save to database
        # if dates exist in outside_of_required range list
        if not_in_required_range:
            # update entry for those dates
            mysql_connect.update_eval_results_entry(results=window_entry_dict,replace_dates=not_in_required_range[0])
            not_in_required_range.pop(0)
        else:
            # otherwise just create a new entry
            mysql_connect.create_eval_results_entry(results=window_entry_dict)

    # delete the outdated stuff that hasnt been replaced
    if not_in_required_range:
        print('         DELETING OUTDATE EVAL RESULTS...')
        for del_dates in not_in_required_range:
            mysql_connect.delete_outdated_eval_results(asset_id=strat_id_metrics.asset_id,
                                                       strat_id=strat_id_metrics.strat_id,
                                                       timeframe=strat_id_metrics.timeframe,
                                                       delete_dates=del_dates)
            print('     ',del_dates)

def select_strategy():
    """runs strategies across all time frames and assets"""

    def get_start_end_dates_dict(dates_df, params):
        dates_df = dates_df.loc[dates_df['strategy_params'] == json.dumps(params)]
        if dates_df.empty:
            return []

        dates_df = dates_df.drop(['strategy_params'], axis=1)
        return dates_df.to_dict(orient='records')

    ##################################################################

    now_date = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)

    # pull active strategies saved in database that are also present here
    active_strategies = mysql_connect.pull_active_strategies_dict()

    select = {'asset':None,'strat':'Basic_Bollinger_Bands_V1'}

    # '2022-03-01'    #'2022-02-01'   for filtering trades list
    date_filter_end, date_filter_start = helper_functions.get_last_x_from_start_of_month(months=BACKTEST_MONTHS)
    start_time = datetime.now()
    for asset_list in MARKETS_BREAKDOWN.values():
        for asset in asset_list:

            if select['asset']:
                if asset != select['asset']:
                    continue

            start_time = datetime.now()
            for timeframe in TIMEFRAMES:
                print(asset)
                asset_id = mysql_connect.pull_all_assets_info(asset_ticker=asset)['asset_id']

                # update asset x for timeframe y #TODO consider handling this separatly, update everything and then run strategies
                mysql_connect.update_asset_ohlcv_data(asset_name=asset,
                                                      timeframe=timeframe)
                # pull candle data
                candle_data_df = mysql_connect.pull_ohlcv_asset_data(asset_name=asset,
                                                                     timeframe=timeframe)

                # make sure data pulled is labeled and organized correctly
                candle_data_df = helper_functions.clean_price_data_DB_to_DF(df=candle_data_df)

                # LOOP THROUGH THE STRATEGIES
                total_perms = 0
                for strategy_name,strategy in active_strategies.items():

                    # if select['strat']:
                    #     if strategy_name != select['strat']:
                    #         continue

                    # pull strategy id
                    strategy_id = mysql_connect.pull_strategy_info(strat_name=strategy_name)
                    strategy_id = strategy_id[0]

                    # get permutations list for strategy and
                    # offset candles (gives a buffer before the start date to so indicators can fully form before the start date)
                    permutations,perm_labels = STRATEGIES_DICT[strategy_name](params='get perms', data_df=None)
                    print(asset)
                    print('     ',strategy_name)
                    print('         num perms:',len(permutations))

                    eval_start_end_dates_df = mysql_connect.pull_evalation_metrics_dates(strat_id=strategy_id,
                                                                                         asset_id=asset_id,
                                                                                         timeframe=timeframe)
                    ### SINGLE ITERATION #####################################################################

                    # for perm in permutations:
                    #     id_metrics = identifier_metrics()
                    #
                    #     id_metrics.strat_id = strategy_id
                    #     id_metrics.asset_id = asset_id
                    #     id_metrics.asset = asset
                    #     id_metrics.timeframe = timeframe
                    #     id_metrics.strategy = strategy_name
                    #     #id_metrics.strategy_params = helper_functions.order_params_dict(strat_params=dict(zip(perm_labels,perm)))
                    #     id_metrics.strategy_params = helper_functions.order_params_dict(strat_params=dict(zip(perm_labels,perm)))# creates ordered dict
                    #     id_metrics.date_added = str(now_date)
                    #
                    #     run_strategy(strat_id_metrics=id_metrics,
                    #                  candle_data=candle_data_df,
                    #                  existing_record_dates=get_start_end_dates_dict(dates_df=eval_start_end_dates_df,
                    #                                                                 params=id_metrics.strategy_params),
                    #                  strategy_func=STRATEGIES_DICT[strategy_name])
                    #
                    #     print()
                    # exit()

                    ### MULTIPROCESSING #####################################################################

                    # loop through permuations

                    from concurrent.futures import ProcessPoolExecutor

                    # use no more than 75 percent of PHYSICAL cores available
                    use_cores = helper_functions.get_core_num(physical=True,pct=1)
                    with ProcessPoolExecutor(max_workers=use_cores) as executor:

                        processes = []

                        for perm in permutations:

                            id_metrics = identifier_metrics()

                            id_metrics.strat_id = strategy_id
                            id_metrics.asset_id = asset_id
                            id_metrics.asset = asset
                            id_metrics.timeframe = timeframe
                            id_metrics.strategy = strategy_name
                            id_metrics.strategy_params = helper_functions.order_params_dict(strat_params=dict(zip(perm_labels, perm)))
                            id_metrics.date_added = str(now_date)

                            # get any existing dates for strat params from df
                            existing_record_dates = get_start_end_dates_dict(dates_df=eval_start_end_dates_df,
                                                                             params=id_metrics.strategy_params)

                            # input existing dates into here
                            proc = executor.submit(run_strategy,
                                                   id_metrics,
                                                   candle_data_df,
                                                   existing_record_dates,
                                                   STRATEGIES_DICT[strategy_name])

                            processes.append(proc)


                        for f in tqdm(concurrent.futures.as_completed(processes)):
                            f.result()


                    #print('         saved')

                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')
                print('all strats for:', asset)
                print('     ',((datetime.now()-start_time).seconds)/60,' minutes')
                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')

if __name__ == "__main__":
    select_strategy()