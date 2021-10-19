
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

logger = logging.getLogger(__name__)

def pivots_points(dataframe: pd.DataFrame, timeperiod=1, levels=6) -> pd.DataFrame:
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
    # R1 = PP + 0.382 * (HIGHprev - LOWprev) ... fibonacci
    data["r1"] = data['pivot'] + 0.382 * (high - low)

    data["rS1"] = data['pivot'] + 0.0955 * (high - low)
    data["rS2"] = data['pivot'] + 0.191 * (high - low)
    data["rS3"] = data['pivot'] + 0.2865 * (high - low)




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


class SYMPHONIK_TLT_v1(IStrategy):
    """
     SYMPHONIK_TLT_v1
     
     La base de la Estrategia es: BlueEyes_MPP_v1
    
    Provando en:
     Miku_1m_5m_CSen444v2_N_1_5 1 millon
    """

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
            'rS1_1d': {},
            'rS2_1d': {},
            'rS3_1d': {},
            'r1_1d': {},
            's1_1d': {},
            'ema440': {},
            'ema88': {},
            'ema20': {},
            'T88': {},
            'T33_30m': {},
            'senkou_b_20_1h': {},
            'tenkan_sen_20_1h': {},
            'senkou_b_200': {},
        },
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
                informative_pairs += [(pair, "1d"),(pair, "1h"),(pair, "30m"),(pair, "5m")]

        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        """
        # dataframe "1d"
        """

        dataframe1d = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1d")

        # Pivots Points
        pp = pivots_points(dataframe1d)
        dataframe1d['pivot'] = pp['pivot']
        dataframe1d['r1'] = pp['r1']
        dataframe1d['s1'] = pp['s1']
        dataframe1d['rS1'] = pp['rS1']
        dataframe1d['rS2'] = pp['rS2']
        dataframe1d['rS3'] = pp['rS3']


        dataframe = merge_informative_pair(
            dataframe, dataframe1d, self.timeframe, "1d", ffill=True)

        """
        # dataframe "1h"
        """
        dataframe1h = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1h")

        create_ichimoku(dataframe1h, conversion_line_period=20, 
                        displacement=9, base_line_periods=88, laggin_span=88)


        dataframe = merge_informative_pair(
            dataframe, dataframe1h, self.timeframe, "1h", ffill=True)

        """
        # dataframe "30m"
        """
        dataframe30m = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="30m")

        dataframe30m['T33'] = ta.T3(dataframe30m, timeperiod=33)

        dataframe = merge_informative_pair(
            dataframe, dataframe30m, self.timeframe, "30m", ffill=True)

        """
        # dataframe normal "5m"
        """
        create_ichimoku(dataframe, conversion_line_period=200, 
                        displacement=88, base_line_periods=200, laggin_span=88)

        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema88'] = ta.EMA(dataframe, timeperiod=88)
        dataframe['ema440'] = ta.EMA(dataframe, timeperiod=440)
        dataframe['T88'] = ta.T3(dataframe, timeperiod=88)




        """
        NOTE: Start Trading

        """

        # Start Trading

        dataframe['trending_start'] = (
            (dataframe['close'] > dataframe['senkou_b_20_1h']) &
            (dataframe['close'] > dataframe['tenkan_sen_20_1h']) &
            (dataframe['tenkan_sen_20_1h'] > dataframe['senkou_a_200']) &
            (dataframe['tenkan_sen_20_1h'] > dataframe['T33_30m']) &
            (dataframe['ema88'] > dataframe['ema440']) &
            (dataframe['ema88'] > dataframe['T88']) &
            (dataframe['ema20'] > dataframe['ema88']) &
            (dataframe['close'] > dataframe['pivot_1d']) &
            (dataframe['close'] < dataframe['rS3_1d'])
        ).astype('int')        

        
        dataframe['trending_over'] = (
            (dataframe['ema88'] < dataframe['T88']) &
            (dataframe['close'] < dataframe['tenkan_sen_20_1h'])
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
