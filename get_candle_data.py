import requests
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from datetime import datetime, timedelta
import pandas as pd
import time
import helper_functions
from settings import MARKETS_BREAKDOWN
import mysql_connect

import os
from dotenv import load_dotenv
load_dotenv()


def update_asset(ticker, timeframe):
    # TODO normalize input params in all of these
    # TODO normalize the output of all these

    """
    Universal Timeframes
    H1 - 1 hour
    H4 - 4 hour
    D1 - 1 day
    ...
    """

    def search_dict(values, searchFor):
        for k in values:
            for v in values[k]:
                if searchFor in v:
                    return k
        return None

    def add_id_cols_to_df(data_df):
        db_obj = mysql_connect.establish_DB_conection()
        cursor = db_obj.cursor()
        cursor.execute("SELECT asset_id FROM assets WHERE asset_ticker = '" + ticker + "'", )
        asset_id = [c[0] for c in cursor]

        if len(asset_id) > 1:
            print('PROBLEM WITH MULTIPLE ASSET IDS')
            exit()

        data_df['asset_ticker_identification'] = asset_id[0]
        data_df['asset_ticker'] = ticker
        data_df['timeframe'] = timeframe
        return data_df

    ########################################################

    def Yahoo_Finance_API(ticker, timeframe):
        """
        https://pypi.org/project/yfinance/
        """
        import yfinance as yf
        timeframes = {'H1': '1H',  # TODO doesnt have a 4 hour option need to correct
                      'D1': '1D'}
        end_date, start_date = helper_functions.get_last_x_from_start_of_month()

        # Get the data
        data_df = yf.download(tickers=ticker,
                              start=start_date,
                              end=end_date,
                              interval=timeframes[timeframe])

        data_df = data_df.reset_index()

        data_df = data_df.rename(columns={'index': 'datetime'})

        data_df['datetime'] = pd.to_datetime(data_df['datetime'])
        data_df['datetime'] = data_df['datetime'].dt.tz_localize(None)

        data_df = data_df.drop(['Adj Close'], axis=1)

        data_df = data_df[(data_df['datetime'] <= datetime.strptime(end_date, '%Y-%m-%d')) &
                          (data_df['datetime'] >= datetime.strptime(start_date, '%Y-%m-%d'))]

        return data_df

    def OANDA_data(ticker, timeframe, num_days=None):
        """
        references
        http://developer.oanda.com/rest-live-v20/instrument-ep/
        http://developer.oanda.com/rest-live-v20/primitives-df/#PricingComponent
        https://oanda-api-v20.readthedocs.io/en/latest/contrib/factories/instrumentscandlesfactory.html

                    Top of the hour alignment
                    “M2” - 2 minutes
                    “M3” - 3 minutes
                    “M4” - 4 minutes
                    “M5” - 5 minutes
                    “M10” - 10 minutes
                    “M15” - 15 minutes
                    “M30” - 30 minutes
                    “H1” - 1 hour

                    Start of day alignment (default 17:00, Timezone/New York)
                    “H2” - 2 hours
                    “H3” - 3 hours
                    “H4” - 4 hours
                    “H6” - 6 hours
                    “H8” - 8 hours
                    “H12” - 12 hours
                    “D” - 1 Day

        instrument = 'EUR_USD'
        granularity = 'M15'
        _from = "2019-09-01T00:00:00Z"
        params = {"from": _from,
                  'granularity': timeframe,
                  'count': 2}

                """

        ############################################################################################

        oanda_key = os.getenv('OANDA_KEY')
        oanda_acct_ID = os.getenv('OANDA_ACCT_ID')
        client = API(access_token=oanda_key, environment='practice')

        ticker = ticker.replace('-', '_')

        spread = 'M'  # cand get B for bid or A for ask or M for mid
        if spread == 'A':
            BAM = 'ask'
        elif spread == 'B':
            BAM = 'bid'
        else:
            BAM = 'mid'

        start_date, end_date = helper_functions.get_last_x_from_start_of_month()

        params = {'granularity': timeframe,
                  # 'count': num_days,
                  'price': spread,
                  'from': end_date+'T00:00:00Z',
                  'to': start_date+'T00:00:00Z'}

        data = []
        for r in InstrumentsCandlesFactory(instrument=ticker, params=params):
            rv = client.request(r)
            if rv['candles']:
                for i in rv['candles']:
                    print(i)

                    candle_time = i['time']
                    candle_time = candle_time.replace('Z', '')
                    candle_time = candle_time.replace('T', ' ')

                    volume = i['volume']
                    open_price = i[BAM]['o']
                    high_price = i[BAM]['h']
                    low_price = i[BAM]['l']
                    close_price = i[BAM]['c']
                    row = [candle_time, open_price, high_price, low_price, close_price, volume]
                    data.append(row)

        df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

        # converts to my time
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'] - timedelta(hours=6)

        return df

    def coinbase_API(ticker, timeframe):

        def get_formated_start_end_times(timeframe_period, end_date):
            """

            takes the given timeframe and end time and provides a start time 300 timeframe periods from end time as per the coinbase api
                \_________________________________________/
              start date                              end date
            """
            # adjust for short datetimes
            end_date = str(end_date)
            if len(end_date) <= 10:
                end_date += " 00:00:00"

            # convert date string to timestamp
            end_date = int(time.mktime(time.strptime(end_date, '%Y-%m-%d %H:%M:%S')))

            start_date = datetime.fromtimestamp(end_date - (int(timeframe_period) * (COINBASE_MAX_RETURNS - 1)))
            end_date = datetime.fromtimestamp(end_date)

            if start_date < datetime.strptime(HARD_START_DATE, '%Y-%m-%d'):
                start_date = datetime.strptime(HARD_START_DATE, '%Y-%m-%d')

            start_date = str(start_date).replace(' ', 'T')
            end_date = str(end_date).replace(' ', 'T')

            return start_date, end_date

        def estimate_iterations(start, end, time_gran):
            start = int(time.mktime(time.strptime(start, '%Y-%m-%d')))
            end = int(time.mktime(time.strptime(end, '%Y-%m-%d')))

            return int(((end - start) / time_gran) / COINBASE_MAX_RETURNS) + 1

        ######################################################################
        import requests

        ticker = ticker.replace('_', '-')

        timeframes = {'M1': 60,
                      'M5': 300,
                      'M15': 900,
                      'H1': 3600,
                      'H6': 21600,  # TODO handle requests for a 4h time period
                      'D1': 86400}

        COINBASE_MAX_RETURNS = 300

        HARD_END_DATE, HARD_START_DATE = helper_functions.get_last_x_from_start_of_month()

        max_iterations = estimate_iterations(start=HARD_START_DATE, end=HARD_END_DATE, time_gran=timeframes[timeframe])

        base_df = pd.DataFrame(columns=['datetime', 'Low', 'High', 'Open', 'Close', 'Volume'])

        # while the earliest date in the dataframe less than the start date of the full range youre trying to get
        # max iterations are a failsafe incase the API returns some wierd dates so im not stuck pinging the server
        it_num = 0
        curr_iter_end = datetime.now()

        while curr_iter_end > datetime.strptime(HARD_START_DATE, '%Y-%m-%d'):
            it_num += 1

            if base_df.empty:
                curr_iter_end = HARD_END_DATE

            # retrieve the dates that are 300x timeframe periods apart
            curr_iter_start, curr_iter_end = get_formated_start_end_times(timeframe_period=timeframes[timeframe],
                                                                          end_date=str(curr_iter_end))

            # pull the data
            url = f"https://api.pro.coinbase.com/products/{ticker}/candles?start={curr_iter_start}&end={curr_iter_end}&granularity={timeframes[timeframe]}"
            headers = {"Accept": "application/json"}
            response = requests.request("GET", url, headers=headers)
            resp = response.json()

            # put into df
            df = pd.DataFrame(resp, columns=['datetime', 'Low', 'High', 'Open', 'Close', 'Volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

            # handle appends
            if base_df.empty:
                base_df = df
            else:
                base_df = base_df.append(df)

            # set the current iter date
            curr_iter_end = base_df['datetime'].min()

            # failsafe to avoid blasting the server
            if it_num > max_iterations:
                break

            time.sleep(0.5)

        #############################################################

        base_df = base_df.drop_duplicates()
        base_df = base_df.reset_index(drop=True)
        base_df = base_df[(base_df['datetime'] <= datetime.strptime(HARD_END_DATE, '%Y-%m-%d')) &
                          (base_df['datetime'] >= datetime.strptime(HARD_START_DATE, '%Y-%m-%d'))]

        return base_df

    #########################################

    sources = {'Forex':OANDA_data,
               'Equities':Yahoo_Finance_API,
               'Crypto':coinbase_API}

    market = search_dict(values=MARKETS_BREAKDOWN,
                         searchFor=ticker)

    # returns datetime + ohlcv data
    data = sources[market](ticker=ticker, timeframe=timeframe)
    data = add_id_cols_to_df(data_df=data)

    return data
