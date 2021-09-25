# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame, Series
import talib.abstract as ta
import logging
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --------------------------------
import math
import pandas as pd
import numpy as np
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes
from technical.util import resample_to_interval, resampled_merge
from numpy.core.records import ndarray

logger = logging.getLogger(__name__)

def PMAX(dataframe, period=10, multiplier=3, length=12, MAtype=1, src=1):  # noqa: C901
    """
    Function to compute PMAX
    Source: https://www.tradingview.com/script/sU9molfV/
    Pinescript Author: KivancOzbilgic
    Args :
        df : Pandas DataFrame with the columns ['date', 'open', 'high', 'low', 'close', 'volume']
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        length: moving averages length
        MAtype: type of the moving average
    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR), ATR (ATR_$period)
            PMAX (pm_$period_$multiplier_$length_$Matypeint)
            PMAX Direction (pmX_$period_$multiplier_$length_$Matypeint)
    """
    import talib.abstract as ta

    df = dataframe.copy()
    mavalue = "MA_" + str(MAtype) + "_" + str(length)
    atr = "ATR_" + str(period)
    df[atr] = ta.ATR(df, timeperiod=period)
    pm = "pm_" + str(period) + "_" + str(multiplier) + "_" + str(length) + "_" + str(MAtype)
    pmx = "pmX_" + str(period) + "_" + str(multiplier) + "_" + str(length) + "_" + str(MAtype)
    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4
    if MAtype == 1:
        df[mavalue] = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        df[mavalue] = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        df[mavalue] = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        df[mavalue] = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        df[mavalue] = VIDYA(df, length=length)
    elif MAtype == 6:
        df[mavalue] = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        df[mavalue] = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        df[mavalue] = vwma(df, length)
    elif MAtype == 9:
        df[mavalue] = zema(df, period=length)
    # Compute basic upper and lower bands
    df["basic_ub"] = df[mavalue] + (multiplier * df[atr])
    df["basic_lb"] = df[mavalue] - (multiplier * df[atr])
    # Compute final upper and lower bands
    df["final_ub"] = 0.00
    df["final_lb"] = 0.00
    for i in range(period, len(df)):
        df["final_ub"].iat[i] = (
            df["basic_ub"].iat[i]
            if (
                df["basic_ub"].iat[i] < df["final_ub"].iat[i - 1]
                or df[mavalue].iat[i - 1] > df["final_ub"].iat[i - 1]
            )
            else df["final_ub"].iat[i - 1]
        )
        df["final_lb"].iat[i] = (
            df["basic_lb"].iat[i]
            if (
                df["basic_lb"].iat[i] > df["final_lb"].iat[i - 1]
                or df[mavalue].iat[i - 1] < df["final_lb"].iat[i - 1]
            )
            else df["final_lb"].iat[i - 1]
        )

    # Set the Pmax value
    df[pm] = 0.00
    for i in range(period, len(df)):
        df[pm].iat[i] = (
            df["final_ub"].iat[i]
            if (
                df[pm].iat[i - 1] == df["final_ub"].iat[i - 1]
                and df[mavalue].iat[i] <= df["final_ub"].iat[i]
            )
            else df["final_lb"].iat[i]
            if (
                df[pm].iat[i - 1] == df["final_ub"].iat[i - 1]
                and df[mavalue].iat[i] > df["final_ub"].iat[i]
            )
            else df["final_lb"].iat[i]
            if (
                df[pm].iat[i - 1] == df["final_lb"].iat[i - 1]
                and df[mavalue].iat[i] >= df["final_lb"].iat[i]
            )
            else df["final_ub"].iat[i]
            if (
                df[pm].iat[i - 1] == df["final_lb"].iat[i - 1]
                and df[mavalue].iat[i] < df["final_lb"].iat[i]
            )
            else 0.00
        )

    # Mark the trend direction up/down
    df[pmx] = np.where((df[pm] > 0.00), np.where((df[mavalue] < df[pm]), "down", "up"), np.NaN)
    # Remove basic and final bands from the columns
    df.drop(["basic_ub", "basic_lb", "final_ub", "final_lb"], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return df

def pivots_points(dataframe: pd.DataFrame, timeperiod=1, levels=3) -> pd.DataFrame:
    """
    Pivots Points
    https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/
    Formula:
    Pivot = (Previous High + Previous Low + Previous Close)/3
    Resistance #1 = (2 x Pivot) - Previous Low
    Support #1 = (2 x Pivot) - Previous High
    Resistance #2 = (Pivot - Support #1) + Resistance #1
    Support #2 = Pivot - (Resistance #1 - Support #1)
    Resistance #3 = (Pivot - Support #2) + Resistance #2
    Support #3 = Pivot - (Resistance #2 - Support #2)
    ...
    :param dataframe:
    :param timeperiod: Period to compare (in ticker)
    :param levels: Num of support/resistance desired
    :return: dataframe
    """

    data = {}

    low = qtpylib.rolling_mean(
        series=pd.Series(index=dataframe.index, data=dataframe["low"]), window=timeperiod
    )

    high = qtpylib.rolling_mean(
        series=pd.Series(index=dataframe.index, data=dataframe["high"]), window=timeperiod
    )

    # Pivot
    data["pivot"] = qtpylib.rolling_mean(series=qtpylib.typical_price(dataframe), window=timeperiod)

    # Resistance #1
    # data["r1"] = (2 * data["pivot"]) - low ... Standard
    # R1 = PP + 0.382 * (HIGHprev - LOWprev) ... fibonacci Tradingview
    data["r1"] = data['pivot'] + 0.382 * (high - low)

    # Resistance #2
    # data["s1"] = (2 * data["pivot"]) - high ... Standard
    # S1 = PP - 0.382 * (HIGHprev - LOWprev) ... fibonacci Tradingview
    data["s1"] = data["pivot"] - 0.382 * (high - low)

    # Calculate Resistances and Supports >1
    for i in range(2, levels + 1):
        prev_support = data["s" + str(i - 1)]
        prev_resistance = data["r" + str(i - 1)]

        # Resitance
        data["r" + str(i)] = (data["pivot"] - prev_support) + prev_resistance

        # Support
        data["s" + str(i)] = data["pivot"] - (prev_resistance - prev_support)

    return pd.DataFrame(index=dataframe.index, data=data)



def create_ichimoku(dataframe, conversion_line_period, displacement, base_line_periods, laggin_span):
    ichimoku = ftt.ichimoku(dataframe,
                            conversion_line_period=conversion_line_period,
                            base_line_periods=base_line_periods,
                            laggin_span=laggin_span,
                            displacement=displacement
                            )
    dataframe[f'tenkan_sen_{conversion_line_period}'] = ichimoku['tenkan_sen']
    dataframe[f'kijun_sen_{conversion_line_period}'] = ichimoku['kijun_sen']
    dataframe[f'senkou_a_{conversion_line_period}'] = ichimoku['senkou_span_a']
    dataframe[f'senkou_b_{conversion_line_period}'] = ichimoku['senkou_span_b']


class Semaphore_1776_v2_4h_ema20_UP_DOWN(IStrategy):
    # La Estrategia es: SymphonIK_Semaphore_v6 (con MACD)... Probando pivot points
    # Semaphore_1776_v2_4h_ema20_UP_DOWN
    # Optimal timeframe for the strategy
    timeframe = '5m'

    # generate signals from the 1h timeframe
    informative_timeframe = '1h'

    # WARNING: ichimoku is a long indicator, if you remove or use a
    # shorter startup_candle_count your results will be unstable/invalid
    # for up to a week from the start of your backtest or dry/live run
    # (180 candles = 7.5 days)
    startup_candle_count = 444  # MAXIMUM ICHIMOKU

    # NOTE: this strat only uses candle information, so processing between
    # new candles is a waste of resources as nothing will change
    process_only_new_candles = True

    minimal_roi = {
        "0": 10,
    }

    plot_config = {
        'main_plot': {
            'pivot_1d': {},
            'r1_1d': {},
            's1_1d': {},
        },
        'subplots': {
            'MACD': {
                'macd_1h': {'color': 'blue'},
                'macdsignal_1h': {'color': 'orange'},
            },
        }
    }

    # WARNING setting a stoploss for this strategy doesn't make much sense, as it will buy
    # back into the trend at the next available opportunity, unless the trend has ended,
    # in which case it would sell anyway.

    # Stoploss:
    stoploss = -0.10

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe)
                             for pair in pairs]
        if self.dp:
            for pair in pairs:
                informative_pairs += [(pair, "1h"), (pair, "4h"), (pair, "1d")]

        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Pares en "1d"
        dataframe1d = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1d")

        # Pivots Points
        pp = pivots_points(dataframe1d)
        dataframe1d['pivot'] = pp['pivot']
        dataframe1d['r1'] = pp['r1']
        dataframe1d['s1'] = pp['s1']

        dataframe = merge_informative_pair(
            dataframe, dataframe1d, self.timeframe, "1d", ffill=True)

        # Pares en 4h
        dataframe4h = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="4h")

        dataframe4h['hma40'] = ftt.hull_moving_average(dataframe4h, 40)

        dataframe = merge_informative_pair(
            dataframe, dataframe4h, self.timeframe, "4h", ffill=True)

        # Pares en 1h
        dataframe1h = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1h")

        dataframe1h['hma148'] = ftt.hull_moving_average(dataframe1h, 148)
        dataframe1h['hma67'] = ftt.hull_moving_average(dataframe1h, 67)
        dataframe1h['hma40'] = ftt.hull_moving_average(dataframe1h, 40)
        

            # MACD
        macd = ta.MACD(dataframe1h, fastperiod=12,
                       slowperiod=26, signalperiod=9)
        dataframe1h['macd'] = macd['macd']
        dataframe1h['macdsignal'] = macd['macdsignal']

        dataframe = merge_informative_pair(
            dataframe, dataframe1h, self.timeframe, "1h", ffill=True)

        # dataframe normal

        create_ichimoku(dataframe, conversion_line_period=20,
                        displacement=88, base_line_periods=88, laggin_span=88)
        create_ichimoku(dataframe, conversion_line_period=380,
                        displacement=633, base_line_periods=380, laggin_span=266)
        create_ichimoku(dataframe, conversion_line_period=12,
                        displacement=88, base_line_periods=53, laggin_span=53)
        create_ichimoku(dataframe, conversion_line_period=9,
                        displacement=26, base_line_periods=26, laggin_span=52)
        create_ichimoku(dataframe, conversion_line_period=6,
                        displacement=26, base_line_periods=16, laggin_span=31)

        dataframe['hma480'] = ftt.hull_moving_average(dataframe, 480)
        dataframe['hma800'] = ftt.hull_moving_average(dataframe, 800)
        dataframe['ema440'] = ta.EMA(dataframe, timeperiod=440)
        dataframe['ema88'] = ta.EMA(dataframe, timeperiod=88)
        
        dataframe['pmax'] = PMAX

        # Start Trading

        dataframe['ichimoku_ok'] = (
            (dataframe['macd_1h'] > dataframe['macdsignal_1h']) &
            (dataframe['kijun_sen_380'] > dataframe['hma148_1h']) &
            (dataframe['kijun_sen_380'] > dataframe['hma40_1h']) &
            (dataframe['kijun_sen_12'] > dataframe['kijun_sen_380']) &
            (dataframe['close'] > dataframe['ema440']) &
            (dataframe['tenkan_sen_12'] > dataframe['senkou_b_6']) &
            (dataframe['senkou_a_6'] > dataframe['senkou_b_6'])
        ).astype('int')        

        dataframe['trending_over'] = (
            (dataframe['hma67_1h'] > dataframe['ema88']) &
            (dataframe['kijun_sen_20'] > dataframe['close'])
        ).astype('int')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.slow_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['ichimoku_ok'] > 0)
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trending_over'] > 0)
            ), 'sell'] = 1
        return dataframe
