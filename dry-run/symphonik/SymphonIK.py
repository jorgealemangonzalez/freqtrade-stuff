# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import logging
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --------------------------------
import pandas as pd
import numpy as np
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes
from technical.util import resample_to_interval, resampled_merge
from pandas.core.base import PandasObject


logger = logging.getLogger(__name__)

def rolling_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_mean(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).mean()
        except Exception as e:
            return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()

# ---------------------------------------------


def rolling_min(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).min()
    except Exception as e:  # noqa: F841
        return pd.Series(series).rolling(window=window, min_periods=min_periods).min()


# ---------------------------------------------

def rolling_max(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).min()
    except Exception as e:  # noqa: F841
        return pd.Series(series).rolling(window=window, min_periods=min_periods).min()


# ---------------------------------------------

def rolling_weighted_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except Exception as e:  # noqa: F841
        return pd.ewma(series, span=window, min_periods=min_periods)

# ---------------------------------------------


def hull_mov(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    ma = (2 * rolling_mean(series, window / 2, min_periods)) - \
        rolling_mean(series, window, min_periods)
    return rolling_mean(ma, np.sqrt(window), min_periods)

def hma(series, window=200, min_periods=None):
    return hull_mov(series, window=window, min_periods=min_periods)

PandasObject.hull_mov = hull_mov


def numpy_rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def numpy_rolling_series(func):
    def func_wrapper(data, window, as_source=False):
        series = data.values if isinstance(data, pd.Series) else data

        new_series = np.empty(len(series)) * np.nan
        calculated = func(series, window)
        new_series[-len(calculated):] = calculated

        if as_source and isinstance(data, pd.Series):
            return pd.Series(index=data.index, data=new_series)

        return new_series

    return func_wrapper


@numpy_rolling_series
def numpy_rolling_mean(data, window, as_source=False):
    return np.mean(numpy_rolling_window(data, window), axis=-1)


@numpy_rolling_series
def numpy_rolling_std(data, window, as_source=False):
    return np.std(numpy_rolling_window(data, window), axis=-1, ddof=1)

"""
 def typical_schiff(bars):
    res = (bars['high'] + bars['low']) / 2.
    return pd.Series(index=bars.index, data=res)
"""
"""
def typical_schiff(bars, series, window=200, min_periods=None):
    ema = rolling_weighted_mean(series, window=window)
    typical = bars['high']
    res = (typical + ema) / 2.
    return pd.Series(index=bars.index, data=res)
"""
"""
def typical_schiff(bars, window=200, min_periods=None):
    ema = rolling_weighted_mean(bars, window=window)
    typical = bars['high']
    res = (typical + ema) / 2.
    return pd.DataFrame(index=bars.index, data=res)
"""
"""
def typical_schiff(dataframe: pd.Dataframe, timeperiod=88):

    ema = ta.EMA(dataframe, timeperiod=timeperiod)
    typical = dataframe['high']
    res = (typical + ema) / 2
    return pd.DataFrame(index=bars.index, data=res)
"""
def typical_schiff(bars, timeperiod=88):

    ema = ta.EMA(bars, timeperiod=timeperiod)
    typical = bars['high']
    res = (typical + ema) / 2
    return pd.Series(index=bars.index, data=res)

def fibonacci_retracements(df, field="close") -> DataFrame:
    # Common Fibonacci replacement thresholds:
    # 1.0, sqrt(F_n / F_{n+1}), F_n / F_{n+1}, 0.5, F_n / F_{n+2}, F_n / F_{n+3}, 0.0
    thresholds = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0]

    window_min, window_max = df[field].min(), df[field].max()
    # fib_levels = [window_min + t * (window_max - window_min) for t in thresholds]

    # Scale data to match to thresholds
    # Can be returned instead if one is looking at the movement between levels
    data = (df[field] - window_min) / (window_max - window_min)

    # Otherwise, we return a step indicator showing the fibonacci level
    # which each candle exceeds
    return data.apply(lambda x: max(t for t in thresholds if x >= t))

# def sma(series, window=200, min_periods=None):
#    return rolling_mean(series, window=window, min_periods=min_periods)

def pivots_points(dataframe: pd.DataFrame,tpe=13, timeperiod=88, levels=3) -> pd.DataFrame:
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

#    ema = qtpylib.rolling_weighted_mean(
#        series=pd.Series(index=dataframe.index, data=dataframe["ema"]), window=timeperiod
#    )

#    ema = qtpylib.rolling_weighted_mean(
#         series=pd.Series(index=dataframe.index), window=13
#    )

    ema = ta.EMA(dataframe, timeperiod=tpe)

    barshigh = (dataframe["high"])

    low = qtpylib.rolling_mean(
        series=pd.Series(index=dataframe.index, data=dataframe["low"]), window=timeperiod
    )

    high = qtpylib.rolling_mean(
        series=pd.Series(index=dataframe.index, data=dataframe["high"]), window=timeperiod
    )

    # Pivot
#    data["pivot"] = qtpylib.rolling_mean((high + ema) / 2)
    data["pivot"] = qtpylib.rolling_mean(series=typical_schiff(dataframe), window=timeperiod)
#    data["pivot"] = qtpylib.rolling_mean((barshigh + ema) / 2)

    # Resistance #1
    # data["r1"] = (2 * data["pivot"]) - low ... Standard
    # R1 = PP + 0.382 * (HIGHprev - LOWprev) ... fibonacci
    data["r1"] = data['pivot'] + 0.382 * (high - low)

    # Resistance #2
    # data["s1"] = (2 * data["pivot"]) - high ... Standard
    # S1 = PP - 0.382 * (HIGHprev - LOWprev) ... fibonacci
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


class SymphonIK(IStrategy):
    # La Estrategia es: Fernando_pivots
    # Semaphore_1776_v2_4h_ema20_UP_DOWN
    # Fernando_pivots
    # Optimal timeframe for the strategy
    timeframe = '5m'

    # generate signals from the 1h timeframe
    informative_timeframe = '1d'

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
                informative_pairs += [(pair, "1d")]

        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Pares en "1d"
        dataframe1d = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1d")
       
        dataframe1d['ema88'] = ta.EMA(dataframe1d, timeperiod=88)
        
        dataframe1d['fibo'] = fibonacci_retracements(dataframe1d)



        # Pivots Points
        pp = pivots_points(dataframe1d)
        dataframe1d['pivot'] = pp['pivot']
        dataframe1d['r1'] = pp['r1']
        dataframe1d['s1'] = pp['s1']

        dataframe = merge_informative_pair(
            dataframe, dataframe1d, self.timeframe, "1d", ffill=True)

        # dataframe normal

        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)

        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)



        # Start Trading

        dataframe['trending_start'] = (
            (dataframe['close'] > dataframe['pivot_1d']) &
            (dataframe['r1_1d'] > dataframe['close']) &
            (dataframe['pivot_1d'] > dataframe['ema20'])
        ).astype('int')        

        dataframe['trending_over'] = (
            (
            (dataframe['high'] > dataframe['r1_1d'])
            )
            |
            (
            (dataframe['pivot_1d'] > dataframe['close'])   
            )
        ).astype('int')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.slow_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['trending_start'] > 0)
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trending_over'] > 0)
            ), 'sell'] = 1
        return dataframe
