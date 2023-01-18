import strategies
# TODO reconcile strategies with strategies db table

STRATEGIES_DICT = {'basic_X_MA_crossover_V1':strategies.basic_X_MA_crossover_V1_ABRIDGED,
                   'basic_X_MA_crossover_V2':strategies.basic_X_MA_crossover_V2_ABRIDGED,
                   'basic_XY_MA_crossover_V1':strategies.basic_XY_MA_crossover_V1_ABRIDGED,
                   'XYZ_MA_crossover_V1':strategies.XYZ_MA_crossover_V1_ABRIDGED,
                   'RVI_crossover_V1':strategies.RVI_crossover_V1_ABRIDGED,
                   'RVI_crossover_V2':strategies.RVI_crossover_V2_ABRIDGED,
                   'Basic_Bollinger_Bands_V1':strategies.Basic_Bollinger_Bands_V1_ABRIDGED,
                   'Intermediate_Bollinger_Bands_V1':strategies.Intermediate_Bollinger_Bands_V1_ABRIDGED,
                   'basic_heiken_ashi_V1':strategies.basic_heiken_ashi_V1_ABRIDGED,
                   'basic_heiken_ashi_V2':strategies.basic_heiken_ashi_V2_ABRIDGED,
                   'basic_heiken_ashi_V3':strategies.basic_heiken_ashi_V3_ABRIDGED
                   }

# STRATEGIES_DICT = {'basic_X_MA_crossover_V1':strategies.basic_X_MA_crossover_V1_ABRIDGED,
#                    'basic_X_MA_crossover_V2':strategies.basic_X_MA_crossover_V2_ABRIDGED,
#                    'basic_XY_MA_crossover_V1':strategies.basic_XY_MA_crossover_V1_ABRIDGED,
#                    'basic_XY_MA_crossover_V2':strategies.basic_XY_MA_crossover_V2_ABRIDGED,
#                    'XYZ_MA_crossover_V1':strategies.XYZ_MA_crossover_V1_ABRIDGED,
#                    'XYZ_MA_crossover_V2':strategies.XYZ_MA_crossover_V2_ABRIDGED,
#                    'RSI_moving_avg_crossover_V1':strategies.RSI_moving_avg_crossover_V1_ABRIDGED,
#                    'RSI_moving_avg_crossover_V2':strategies.RSI_moving_avg_crossover_V2_ABRIDGED,
#                    'RVI_crossover_V1':strategies.RVI_crossover_V1_ABRIDGED,
#                    'RVI_crossover_V2':strategies.RVI_crossover_V2_ABRIDGED,
#                    'RVI_crossover_V3':strategies.RVI_crossover_V3_ABRIDGED,
#                    'basic_MACD_V1':strategies.basic_MACD_V1_ABRIDGED,
#                    'basic_MACD_V2':strategies.basic_MACD_V2_ABRIDGED,
#                    'basic_MACD_V3':strategies.basic_MACD_V3_ABRIDGED,
#                    'basic_MACD_V4':strategies.basic_MACD_V4_ABRIDGED,
#                    'MA_divergence_V1':strategies.MA_divergence_V1_ABRIDGED,
#                    'Basic_Bollinger_Bands_V1':strategies.Basic_Bollinger_Bands_V1_ABRIDGED,
#                    'Basic_Bollinger_Bands_V2':strategies.Basic_Bollinger_Bands_V2_ABRIDGED,
#                    'Basic_Bollinger_Bands_V3':strategies.Basic_Bollinger_Bands_V3_ABRIDGED,
#                    'Intermediate_Bollinger_Bands_V1':strategies.Intermediate_Bollinger_Bands_V1_ABRIDGED,
#                    'Intermediate_Bollinger_Bands_V2':strategies.Intermediate_Bollinger_Bands_V2_ABRIDGED,
#                    'basic_heiken_ashi_V1':strategies.basic_heiken_ashi_V1_ABRIDGED,
#                    'basic_heiken_ashi_V2':strategies.basic_heiken_ashi_V2_ABRIDGED,
#                    'basic_heiken_ashi_V3':strategies.basic_heiken_ashi_V3_ABRIDGED}

MARKETS_BREAKDOWN = {'Forex':['EUR-USD','USD-JPY','GBP-USD'],  #TODO reconcile assets in assets table
                    'Equities':['SPY','IWM'],
                    'Crypto':['BTC-USD','ETH-USD']}


TIMEFRAMES = ['H1']  # TODO go off oanda timeframes
                     # '1_day','4_hour','30_min','15_min','5_min'
                     #TODO might need to make a timeframes table to scale, read in timeframe from timeframes table, and loop through also create a timeframes table

#translates a timeframe to minutes (use to add offset candles to dates)
TIMEFRAME_TO_MINUTE_TRANSLATIONS = {'H1':60,
                                    'H4':240,
                                    'D1':1440}

TOTAL_MONTHS_OF_DATA = 3
BACKTEST_MONTHS = 2
BUFFER_BARS = 100 # TODO implement this somehow
                  #number of bars to provide a buffer so backtesting can start on the selected date

