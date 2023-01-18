import pandas as pd
import numpy as np
##############################################

def average_true_range(df, window=14):
    window = int(window)
    TR = df['High'] - df['Low']
    df[str(window)+'ATR'] = TR.ewm(span=window).mean()

    return df

def standard_dev(df,window):
    window = int(window)
    df['std_dev_'+str(window)] = df['Close'].rolling(window).std()
    return df

def autocorrelation(df,lag,window):
    """https://www.statology.org/autocorrelation-python/"""

    lag = int(lag)
    window = int(window)

    df['auto_correl_'+str(lag)] = 0

    for idx in df.index:
        if idx < window+lag:
            continue
        orig_slice = df['Close'].iloc[(idx-window):idx].tolist()
        lag_slice = df['Close'].iloc[(idx-window)-lag:(idx-lag)].tolist()

        correl = np.corrcoef(orig_slice, lag_slice)[0, 1]

        df.loc[idx,'auto_correl_'+str(lag)] = correl

    return df

def simple_moving_average(df, MA_num):
    """
    puts simple moving average into dataframe as header ex '9SMA'
    """
    MA_num = int(MA_num)
    # if the MA_num is too big for the current length of the dataframe, print an error and exit
    if MA_num > len(df.index):
        print('ERROR: SMA number is larger than the number of rows in df')
        exit()

    df[str(MA_num)+'SMA'] = df['Close'].rolling(MA_num).mean()

    return df

def exponential_moving_average(df, MA_num):
    """
    puts simple moving average into dataframe as header ex '9EMA'
    """
    MA_num = int(MA_num)
    EMA_hdr = str(MA_num) + 'EMA'

    sma = df['Close'].rolling(window=MA_num, min_periods=MA_num).mean()[:MA_num]
    rest = df['Close'][MA_num:]
    df[EMA_hdr] = pd.concat([sma, rest]).ewm(span=MA_num, adjust=False).mean()
    return df

def bollinger_bands(df, window, std_dev):

    window = int(window)

    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()

    df['BB mean'] = rolling_mean
    df['BB lower'] = rolling_mean - (rolling_std * std_dev)
    df['BB upper'] = rolling_mean + (rolling_std * std_dev)

    return df

def heiken_ashi(df):

    def red_green(dataframe):
        if dataframe['HA Open'] >= dataframe['HA Close']:
            return 'red'
        elif dataframe['HA Close'] > dataframe['HA Open']:
            return 'green'

    ######################################################################

    df['HA Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close'])/4
    df['HA Open'] = (df['Open'].shift(1) + df['Close'].shift(1))/2

    df['HA High'] = df[['HA Open', 'HA Close', 'High']].max(axis=1)
    df['HA Low'] = df[['HA Open', 'HA Close', 'Low']].min(axis=1)

    df['HA_red_green'] = df.apply(red_green, axis=1)

    return df

def ichimoku(df,tenken_val=9,kijun_val=26,senkou_B_val=52,offset=26):
    """
    tenkan = conversion line
    kijun = base line
    chikou = lagging span
    senkou A = leading span A
    senkou B = leading span B
    kumo = cloud

    not supposed to be adjustable but we'll see in testing
    """
    tenken_val = int(tenken_val)
    kijun_val = int(kijun_val)
    senkou_B_val = int(senkou_B_val)
    offset = int(offset)

    tenkan_high = df['High'].rolling(tenken_val).max()
    tenkan_low = df['Low'].rolling(tenken_val).min()

    df['Tenkan'] = (tenkan_high + tenkan_low) / 2

    kijun_high = df['High'].rolling(kijun_val).max()
    kijun_low = df['Low'].rolling(kijun_val).min()

    df['Kijun'] = (kijun_high + kijun_low) / 2

    # has offset
    df['Projected Senkou A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(offset)
    df['Projected Senkou B'] = ((df['High'].rolling(senkou_B_val).max() + df['Low'].rolling(senkou_B_val).min()) / 2).shift(offset)

    # no offset
    df['Current Senkou A'] = ((df['Tenkan'] + df['Kijun']) / 2)
    df['Current Senkou B'] = ((df['High'].rolling(senkou_B_val).max() + df['Low'].rolling(senkou_B_val).min()) / 2)

    #df['Chikou'] = df['Close'].shift(-26)
    df['Chikou'] = df['Close']

    # offset only for comparison with chikou
    # (so offset * 2 into the future to compare current chikou with what cloud would be offset periods in the past)
    df['Chikou adj Senkou A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(offset*2)
    df['Chikou adj Senkou B'] = ((df['High'].rolling(senkou_B_val).max() + df['Low'].rolling(senkou_B_val).min()) / 2).shift(offset*2)

    return df

def RSI(df, periods):

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=periods - 1, adjust=False).mean()
    ema_down = down.ewm(com=periods - 1, adjust=False).mean()
    rs = ema_up / ema_down

    df[str(periods) + 'RSI'] = 100 - (100 / (1 + rs))

    # Skip first x days to have real values
    df = df.iloc[periods:]

    return df

def RVI(df, periods,signal_MA):
    """relative vigor index"""

    RVI_header = str(periods) + 'RVI'

    a = df['Close'] - df['Open']
    b = df['Close'] - df['Open'].shift(1)
    c = df['Close'] - df['Open'].shift(2)
    d = df['Close'] - df['Open'].shift(3)
    e = df['High'] - df['Low']
    f = df['High'] - df['Low'].shift(1)
    g = df['High'] - df['Low'].shift(2)
    h = df['High'] - df['Low'].shift(3)

    numerator = (a+(2*b)+(2*c)+d)/6
    denominator = (e+(2*f)+(2*g)+h)/6

    numerator_SMA = numerator.rolling(periods).mean()
    denominator_SMA = denominator.rolling(periods).mean()

    df[RVI_header] = numerator_SMA/denominator_SMA

    df[RVI_header+'_signal'] = df[RVI_header].rolling(signal_MA).mean()

    return df

def MACD(df, slow_MA, fast_MA, signal):
    """
    1.Calculate a 12-period EMA of the price for the chosen time period.
    2.Calculate a 26-period EMA of the price for the chosen time period.
    3.Subtract the 26-period EMA from the 12-period EMA.
    4.Calculate a nine-period EMA of the result obtained from step 3.

    defaults are:
    slow: 26
    fast: 12
    signal: 9

    """
    # calculate slow and fast MAs
    df = exponential_moving_average(df=df,MA_num=slow_MA)
    df = exponential_moving_average(df=df,MA_num=fast_MA)

    # get the difference between them for the MACD
    df['MACD'] = df[str(slow_MA) + 'EMA'] - df[str(fast_MA) + 'EMA']

    # get the X EMA of the MACD diff
    df[str(signal)+'signal_line'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    # get the histogram data as diff between MACD and signal line
    df['MACD_hist'] = df['MACD'] - df[str(signal)+'signal_line']

    return df

def rejection_candle_indicator(df, x_candles):
    """
    gives a reading for rejection candles/candle patterns it sees
    is based off length of wicks and weighted by relative volume and relative candle length
    """

    def get_wick_perc():

        OC_max = df[['Open','Close']].max(axis=1)
        OC_min = df[['Open','Close']].min(axis=1)

        upper_wick_length = df['High'] - OC_max  # signals rejection to the upside
        lower_wick_length = OC_min - df['Low']  # signals rejection to the downside

        # upper = upper_wick_length/(max(df['Close'],df['Open']) - df['Low']) # upper wick as a percentage of max of open/close to the low
        # lower = lower_wick_length/(df['High'] - min(df['Close'],df['Open'])) # lower wick as a percentage of min of open/close to the high

        HL = df['High'] - df['Low']

        upper = upper_wick_length / HL
        lower = lower_wick_length / HL

        return upper,lower

    def get_body_retrace():
        open_overlap = (df['Open'] - df['Open'].shift(1)) / (df['Close'].shift(1) - df['Open'].shift(1))
        close_overlap = (df['Close'] - df['Open'].shift(1)) / (df['Close'].shift(1) - df['Open'].shift(1))
        return open_overlap - close_overlap

    ##################################################################################

    upper_wick_perc, lower_wick_perc = get_wick_perc()
    candle_length = abs(df['High'] - df['Low'])
    #body_length = abs(df['Close'] - df['Open']) # signals degree of momentum
    relative_volume = (df['Volume']/df['Volume'].rolling(x_candles).mean()) + 1 # signals how important the current candle is
    #relative_body_length = body_length/body_length.rolling(x_candles).mean() # signals how much momentum is involved in the current candle
    relative_candle_length = candle_length/candle_length.rolling(x_candles).mean() # signals how much momentum is involved in the current candle
    #body_retrace = get_body_retrace() # measures how much the current candle body retraced the previous one (positive = retrace, negatvie = extension)

    # calculate signals from upper/lower wick perc and body retrace
    # wick readings
    df['wick_readings'] = (upper_wick_perc - lower_wick_perc) * relative_volume * relative_candle_length

    # amplify the signal readings by perceived importance from relative volume and candle/body length
    return df

def swing_HL(df, window_size):
    """
    WILL LIKELY NEED TO BE ON A PER ITERATION BASIS

    maps out the most recent swing HL point given a certain window size
    - start with rolling window
    - get idx of the max/min OC/HL price in that window
    - create another rolling window for the idx numbers, if all idx numbers in that rolling window are the same, that is the swing HL point
    """

    def rolling_idx_max(series):
        max_idx = series.idxmax()
        return max_idx

    def rolling_idx_min(series):
        max_idx = series.idxmin()
        return max_idx

    def get_HLs(series_name,target_column):
        series = df[series_name]
        series = series.dropna()
        series = series.astype(int)

        HL_list = []

        window = []
        for i in series:
            try:
                window.append(int(i))
            except:
                pass

            if len(window) < window_size:
                continue

            if all(x == window[0] for x in window):
                HL_list.append([df.iloc[i]['Date_Time'],df.iloc[i][target_column]])

            del window[0]

        return HL_list

    df['body high'] = df[['Open','Close']].max(axis=1)
    df['body low'] = df[['Open','Close']].min(axis=1)
    df['wick high'] = df['High']
    df['wick low'] = df['Low']

    df['body window idxmax'] = df['body high'].rolling(window_size).apply(rolling_idx_max)
    df['body window idxmin'] = df['body low'].rolling(window_size).apply(rolling_idx_min)

    df['wick window idxmax'] = df['High'].rolling(window_size).apply(rolling_idx_max)
    df['wick window idxmin'] = df['Low'].rolling(window_size).apply(rolling_idx_min)

    body_high_df = pd.DataFrame((get_HLs(series_name='body window idxmax',target_column='body high')),columns=['Date_Time','body_swing_high'])
    body_low_df = pd.DataFrame((get_HLs(series_name='body window idxmin', target_column='body low')),columns=['Date_Time', 'body_swing_low'])
    wick_high_df = pd.DataFrame((get_HLs(series_name='wick window idxmax',target_column='wick high')),columns=['Date_Time','wick_swing_high'])
    wick_low_df = pd.DataFrame((get_HLs(series_name='wick window idxmin', target_column='wick low')),columns=['Date_Time', 'wick_swing_low'])

    df = df.merge(body_high_df, on='Date_Time', how='left')
    df = df.merge(body_low_df, on='Date_Time', how='left')
    df = df.merge(wick_high_df, on='Date_Time', how='left')
    df = df.merge(wick_low_df, on='Date_Time', how='left')

    cols_to_remove = ['body window idxmax','body window idxmin','wick window idxmax','wick window idxmin','body high',
                      'body low','wick high','wick low']

    df = df.drop(cols_to_remove, axis=1)
    #df = df.filter(['Date_Time','Open','High','Low','Close','Volume','body_swing_high','body_swing_low','wick_swing_high','wick_swing_low'])

    return df

def swing_HL_std_dev(df, HL_window_size, std_dev_window):
    """
    WILL LIKELY NEED TO BE ON A PER ITERATION BASIS

    retrieves the std deviation between price differences of sequential highs/lows
    gets most recent high and low (wicks and bodies), gets price difference, calculates rolling std dev
    """

    def get_swing_lengths(df):

        swing_points = []

        for idx in df.index:
            curr_row = df.loc[idx]
            date = curr_row['Date_Time']
            high = curr_row['swing_high']
            low = curr_row['swing_low']

            # make appends to list
            if high:
                swing_points.append([date,high,'high'])
            if low:
                swing_points.append([date,low,'low'])

            # # make sure theres a high and a low in the list
            # if not list_filled:
            #     if any('high' in a for a in latest_point) and any('low' in a for a in latest_point):
            #         list_filled = True
            #
            # # look backwards to the last two points in the list, if theyre opposite, record the distance
            # if latest_point[-1][2] != latest_point[-2][2]:
            #     price_difs.append(abs(latest_point[-1][1]-latest_point[-2][1]))

        price_difs = []
        same_points = []



        for enum in range(0,len(swing_points)):
            if enum == 0:
                continue

            curr = swing_points[enum]
            prev = swing_points[enum-1]

            # compare the last two points
            # if different calc differnce and append
            if curr[2] != prev[2]:
                # if same points list exists
                if same_points:

                    # loop through same points and get differences compared to latest point
                    for same_point in same_points:
                        price_difs.append([same_point[0],curr[0],abs(same_point[1]-curr[1])])

                    same_points = []
                    print()
                    continue
                else:
                    price_difs.append([prev[0],curr[0],abs(curr[1] - prev[1])])
                    print()

            # if same append the two points to same_points
            elif curr[2] == prev[2]:
                if curr[2] == 'high':
                    if curr[1] >= prev[1]:
                        continue

                if curr[2] == 'low':
                    if curr[1] <= prev[1]:
                        continue

                if curr not in same_points:
                    same_points.append(curr)

                if prev not in same_points:
                    same_points.append(prev)

                """
                    for multiple highs:
                    put swing points in chrono order -> for p0 if p1 to px is > p0, delete p0 from the list

                    for multiple lows
                    put swing points in chrono order -> for p0 if p1 to px is < p0, delete p0 from the list
                    """

        return price_difs

    #################################################################################

    df = swing_HL(df=df,window_size=HL_window_size)

    body_HL_df = df[['Date_Time','body_swing_high','body_swing_low']].copy()
    body_HL_df.rename(columns={'body_swing_high':'swing_high','body_swing_low':'swing_low'}, inplace=True)

    #wick_HL_df = df[['Date_Time','wick_swing_high','wick_swing_low']].copy()

    body_HL_df = body_HL_df.dropna(thresh=2)
    body_HL_df = body_HL_df.fillna(False)
    #wick_HL_df = wick_HL_df.dropna(thresh=2)

    get_swing_lengths(df=body_HL_df)

def SBV_flow(df):
    """https://www.marketvolume.com/technicalanalysis/sbvflow.asp
    other volumne indicators
    https://patternswizard.com/technical-analysis/indicators/volume-indicators/
    """
    return df

def efficiency_ratio(df,window_size):
    """
    https://stockbee.blogspot.com/2007/05/how-to-find-stocks-with-smooth-trends.html
    can use efficiency ratio as a filter for other indicators, i.e. if efficiency is low then ignore indicators, if high then take indicators more seriously
    """
    df['direction'] = df['Close'].diff(window_size).abs()
    df['volatility'] = df['Close'].diff(1).abs()

    df['Efficiency_Ratio'] = df['direction']/df['volatility'].rolling(window_size).sum()

    df = df.drop(['direction','volatility'],axis=1)

    return df

def money_flow_index(df,window_size):
    """
    """

    df['raw_MF'] = ((df['High']+df['Low']+df['Close'])/3) * df['Volume']

    df['dir_MF_pos'] = np.where(df['Close'] > df['Open'], df['raw_MF'], 0)
    df['dir_MF_neg'] = np.where(df['Close'] < df['Open'], df['raw_MF']*-1, 0)

    df['rolling_pos_sum'] = df['dir_MF_pos'].rolling(window_size).sum()
    df['rolling_neg_sum'] = df['dir_MF_neg'].rolling(window_size).sum()

    df['pos_neg_ratio'] = abs(df['rolling_pos_sum'])/abs(df['rolling_neg_sum'])

    df['money_flow_index'] = 100 - 100/(1+df['pos_neg_ratio'])

    df = df.drop(['raw_MF','dir_MF_pos','dir_MF_neg','rolling_pos_sum','rolling_neg_sum','pos_neg_ratio'],axis=1)
    return df

def MA_price_divergence(df,MA_num, use_SMA=1):

    if use_SMA == 1:
        df = simple_moving_average(df=df,MA_num=MA_num)
    else:
        df = exponential_moving_average(df=df,MA_num=MA_num)

    df['MA_Price_Divergence'] = 3

    return df
