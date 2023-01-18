import mysql.connector
import pandas as pd
import numpy
import get_candle_data
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import helper_functions

import os
from dotenv import load_dotenv
load_dotenv()

# CONNECTIONS
def establish_DB_conection():
    mydb = mysql.connector.connect(host='localhost',
                                   user='root',
                                   passwd=os.getenv('DB_PWD'),
                                   database='django_backtesting_db',
                                   auth_plugin='mysql_native_password')

    return mydb

def establish_SQLAlchemy_connection():
    db_data = 'mysql+mysqlconnector://' + 'root' + ':' + os.getenv('DB_PWD') + '@' + 'localhost' + ':3306/'+'django_backtesting_db'\
              + '?auth_plugin=mysql_native_password'
    return create_engine(db_data)

#########################################################

def pull_strategy_info(strat_name):

    query = "SELECT * FROM strategies WHERE strategy_reference='"+strat_name+"';"
    db_obj = establish_DB_conection()
    cursor = db_obj.cursor()

    cursor.execute(query)

    ret_val = None
    for s in cursor:
        if s[1] == strat_name:
            ret_val = s
            break

    cursor.close()
    db_obj.close()
    return ret_val

def pull_active_strategies_dict():
    """
    lines up what you have in the database vs the strats dict
    """
    from settings import STRATEGIES_DICT
    db_obj = establish_DB_conection()
    cursor = db_obj.cursor()

    query = "SELECT * FROM strategies;"

    ret_strats_dict = {}

    cursor.execute(query)
    for s in cursor:
        ret_strats_dict[s[1]] = STRATEGIES_DICT[s[1]]

    cursor.close()
    db_obj.close
    return ret_strats_dict

def pull_all_assets_info(asset_ticker):
    query = "SELECT * FROM assets;"

    db_obj = establish_DB_conection()
    cursor = db_obj.cursor(dictionary=True)

    cursor.execute(query)
    res_dict = cursor.fetchall()
    ret_val = None
    for a in res_dict:
        if a['asset_ticker'] == asset_ticker:
            ret_val = a
            break

    cursor.close()
    db_obj.close()
    return ret_val

def pull_ohlcv_asset_data(asset_name, timeframe):
    db_obj = establish_DB_conection()
    cursor = db_obj.cursor()

    query = "SELECT * FROM asset_ohlcv WHERE asset_ticker = '" + asset_name + "' AND timeframe = '" + timeframe + "'"

    df = pd.read_sql(query, con=db_obj)

    cursor.close()
    db_obj.close()

    return df

# handles updates for specific asset
#TODO sync the timezone info because its screwing up the trade data
def update_asset_ohlcv_data(asset_name,timeframe):
    """
    inputs asset name and timeframe
    checks to see if data needs an update
    if it does delete whats there and pull data
    """

    from settings import TOTAL_MONTHS_OF_DATA

    def delete_all_ohlcv_data():
        """
        delete any data older than TOTAL_MONTHS_OF_DATA months from current month
        """
        print('     DELETING OLD DATA FOR',asset_name,':',timeframe)

        db_obj = establish_DB_conection()
        cursor = db_obj.cursor()

        query = "DELETE FROM asset_ohlcv WHERE asset_ticker = '" + asset_name + "' AND timeframe = '" + timeframe + "'"

        cursor.execute(query)

        cursor.close()
        db_obj.commit()
        db_obj.close()
        print('         ...DELETE COMPLETED')

    #####################################################################
    print('UPDATING DATA FOR - ',asset_name,':',timeframe)

    df = pull_ohlcv_asset_data(asset_name=asset_name,timeframe=timeframe)

    df['datetime'] = pd.to_datetime(df['datetime'])
    newest_date = df['datetime'].max()
    oldest_date = df['datetime'].min()

    first_of_month = datetime.today().replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    # if df is empty, run the update
    if df.empty:
        print('OHLCV table empty, pulling data for',asset_name)
        df = get_candle_data.update_asset(ticker=asset_name,
                                          timeframe=timeframe)
        engine = establish_SQLAlchemy_connection()
        df.to_sql('asset_ohlcv', engine, if_exists='append', index=False)
        engine.dispose()

    #if the newest time series date compared to the first of the month are more than a day apart it needs an update

    elif (first_of_month - newest_date).days > 30:
        print('OHLCV table outdated, pulling data for', asset_name,':',timeframe)

        # delete whats there and get all new data
        delete_all_ohlcv_data()

        # pull data from source for x months
        df = get_candle_data.update_asset(ticker=asset_name,
                                          timeframe=timeframe)

        engine = establish_SQLAlchemy_connection()
        df.to_sql('asset_ohlcv', engine, if_exists='append', index=False)
        engine.dispose()

    else:
        print('DATA DOESNT NEED UPDATE')

def delete_outdated_eval_results(strat_id, asset_id, timeframe, delete_dates):
    """
    checks eval metrics for existing stats and deletes any that are outdated
    ie more than BACKTEST_MONTHS old
    """
    from settings import BACKTEST_MONTHS

    _,cutoff_date = helper_functions.get_last_x_from_start_of_month(months=BACKTEST_MONTHS)

    db_obj = establish_DB_conection()
    cursor = db_obj.cursor()

    query = "DELETE FROM evaluation_metrics " \
            "WHERE strat_id = '" + str(strat_id) + "' AND" \
                 " asset_id = '" + str(asset_id) + "' AND" \
                 " timeframe = '" + timeframe + "' AND" \
                 " period_start_date = '" + delete_dates['month_start'] + "' AND" \
                 " period_end_date = '" + delete_dates['month_end'] + "'"

    cursor.execute(query)

    cursor.close()
    db_obj.commit()
    db_obj.close()

def update_eval_results_entry(results,replace_dates):
    db_obj = establish_DB_conection()
    cursor = db_obj.cursor()

    ignore = ['strat_id','asset_id','asset','timeframe','strategy','strategy params','date_added']

    set_dict = {}
    for k,v in results.items():
        if k not in ignore:
            set_dict[k] = v

    query_base = "UPDATE evaluation_metrics SET "
    query_where = " WHERE strat_id = {} AND" \
                  " asset_id = {} AND" \
                  " period_end_date = '{}' AND" \
                  " period_start_date = '{}' AND" \
                  " timeframe = '{}' AND" \
                  " strategy_params = '{}'".format(results['strat_id'],
                                                   results['asset_id'],
                                                   replace_dates['month_end'], # TODO replace start and end dates with the dates that need to be replaced
                                                   replace_dates['month_start'],
                                                   results['timeframe'],
                                                   results['strategy_params'])

    set_cols = []
    insert_tup = []

    for enum, (k, v) in enumerate(results.items()):
        if k not in ignore:
            set_cols.append(str(k) + ' = %s')

            # handle nans
            try:
                if numpy.isnan(v):
                    v = None
            except:
                pass
            # convert numpy floats to floats
            if isinstance(v, numpy.float64):
                v = float(v)

            insert_tup.append(v)


    set_cols = ', '.join(set_cols)
    insert_tup = tuple(insert_tup)

    total_query = query_base + set_cols + query_where

    cursor.execute(total_query,insert_tup)

    db_obj.commit()

    cursor.close()
    db_obj.close()
    return

def create_eval_results_entry(results):

    db_obj = establish_DB_conection()
    cursor = db_obj.cursor()

    query_base = "INSERT INTO evaluation_metrics "
    query_cols = []
    query_ph = []
    insert_tup = []

    try:
        ignore = ['asset','date_added','strategy']
        for enum, (k, v) in enumerate(results.items()):
            if k not in ignore:
                query_cols.append(str(k))
                query_ph.append('%s')

                # handle nans
                try:
                    if numpy.isnan(v):
                        v = None
                except:
                    pass
                # convert numpy floats to floats
                if isinstance(v,numpy.float64):
                    v = float(v)

                insert_tup.append(v)

        query_cols = ','.join(query_cols)
        query_ph = ','.join(query_ph)
        insert_tup = tuple(insert_tup)

        total_query = query_base + '(' + query_cols + ') VALUES (' + query_ph + ')'
        cursor.execute(total_query, insert_tup)

        db_obj.commit()
    except Exception as E:
        print(E)
        print('insert tup:', insert_tup)

    cursor.close()
    db_obj.close()

def pull_evalation_metrics_dates(strat_id,asset_id,timeframe):
    db_obj = establish_DB_conection()
    cursor = db_obj.cursor()

    query = "SELECT strategy_params, period_end_date, period_start_date " \
            "FROM evaluation_metrics " \
            "WHERE strat_id = '" + str(strat_id) + "' AND asset_id = '" + str(asset_id) + "' AND timeframe = '" + str(timeframe) + "'"

    df = pd.read_sql(query, con=db_obj)
    df = df.rename(columns={'period_end_date':'month_end','period_start_date':'month_start'})

    cursor.close()
    db_obj.close()

    return df








