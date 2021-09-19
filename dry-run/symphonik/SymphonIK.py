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


# Obelisk_Ichimoku_Slow v1.3 - 2021-04-20
#
# by Obelisk
# https://github.com/brookmiles/
#
# 1.3 increase return without too much additional drawdown
#  - add ema entry guards
#  - remove cloud top exit signal from 1.2
#
# The point of this strategy is to buy and hold an up trend as long as possible.
# If you are tempted to add ROI or trailing stops, you will need to make other modifications as well.
#
# This strategy can optionally be backtested at 5m or 1m to validate roi/trailing stop behaviour (there isn't any).
#
# WARNING
#
# Do not use stoploss_on_exchange or the bot may trigger emergencysell when it
# fails to place the stoploss.
#
# WARNING
#
# This strat will buy into ongoing trends, so pay attention to timing when you launch it.
# If the market is peaking then you may end up buying into trends that are just about to end.
#
#
# Contributions:
# JimmyNixx - SSL Channel confirmation
#
# Backtested with pairlist generated with:
#     "pairlists": [
#         {
#             "method": "VolumePairList",
#             "number_assets": 25,
#             "sort_key": "quoteVolume",
#             "refresh_period": 1800
#         },
#         {"method": "AgeFilter", "min_days_listed": 10},
#         {"method": "PrecisionFilter"},
#         {"method": "PriceFilter", "low_price_ratio": 0.001},
#         {
#             "method": "RangeStabilityFilter",
#             "lookback_days": 3,
#             "min_rate_of_change": 0.1,
#             "refresh_period": 1440
#         },
#     ],

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


def weigted_sum(window_data, weights, averager):
    weighted_sum_window = weights*window_data
    weighted_sum = np.sum(weighted_sum_window)
    result =  weighted_sum / averager
    return result


def wma(series, window):
    weights = np.arange(1, window+1)
    averager = window * (window +1) / 2
    return series.rolling(window).apply(lambda x: weigted_sum(x, weights, averager))


def hma(series, window):
    series = series['close']
    ma = (2 * wma(series, int(round(window / 2)))) - \
        wma(series, window)
    return wma(ma, int(round(np.sqrt(window))))


class SymphonIK(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '5m'

    # generate signals from the 5m timeframe
    informative_timeframe = '1h'

    # WARNING: ichimoku is a long indicator, if you remove or use a
    # shorter startup_candle_count your results will be unstable/invalid
    # for up to a week from the start of your backtest or dry/live run
    # (180 candles = 7.5 days)
    startup_candle_count = 888  # for hma888

    # NOTE: this strat only uses candle information, so processing between
    # new candles is a waste of resources as nothing will change
    process_only_new_candles = True

    minimal_roi = {
        "0": 10,
    }

    plot_config = {
        'main_plot': {
            'kijun_sen_380': {},
            'kijun_sen_12': {},
            'hma148_1h': {},
            'hma40_1h': {},
            'ema440': {},
            'tenkan_sen_12': {},
            'senkou_a_6': {},
            'senkou_b_6': {},
            'hma67_1h': {},
            'ema88': {},
            'kijun_sen_20': {},
            'close': {
                'color': 'black',
            },
        },
        'subplots': {
            'MACD': {
                'macd_2h': {'color': 'blue'},
                'macdsignal_2h': {'color': 'orange'},
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
                informative_pairs += [(pair, "1h"), (pair, "2h")]

        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
                # Pares en 1h
        dataframe1h = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1h")

        dataframe1h['hma148'] = ftt.hull_moving_average(dataframe1h, 148)
        dataframe1h['hma67'] = ftt.hull_moving_average(dataframe1h, 67)
        dataframe1h['hma40'] = ftt.hull_moving_average(dataframe1h, 40)

        dataframe = merge_informative_pair(
            dataframe, dataframe1h, self.timeframe, "1h", ffill=True)

        # Pares en 2h
        dataframe2h = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="2h")

        # MACD
        macd = ta.MACD(dataframe2h, fastperiod=12,
                       slowperiod=26, signalperiod=9)
        dataframe2h['macd'] = macd['macd']
        dataframe2h['macdsignal'] = macd['macdsignal']

        dataframe = merge_informative_pair(
            dataframe, dataframe2h, self.timeframe, "2h", ffill=True)

        # dataframe normal

        create_ichimoku(dataframe, conversion_line_period=380,
                        displacement=633, base_line_periods=380, laggin_span=266)
        create_ichimoku(dataframe, conversion_line_period=12,
                        displacement=88, base_line_periods=53, laggin_span=53)
        create_ichimoku(dataframe, conversion_line_period=20,
                        displacement=88, base_line_periods=88, laggin_span=88)
        create_ichimoku(dataframe, conversion_line_period=9,
                        displacement=26, base_line_periods=26, laggin_span=52)
        create_ichimoku(dataframe, conversion_line_period=6,
                        displacement=26, base_line_periods=16, laggin_span=31)

        dataframe['hma480'] = hma(dataframe, 480)
        dataframe['hma800'] = hma(dataframe, 800)
        dataframe['ema440'] = ta.EMA(dataframe, timeperiod=440)
        dataframe['ema88'] = ta.EMA(dataframe, timeperiod=88)

        dataframe['ichimoku_ok'] = (
            (dataframe['kijun_sen_380'] > dataframe['hma148_1h']) &
            (dataframe['kijun_sen_380'] > dataframe['hma40_1h']) &
            (dataframe['kijun_sen_12'] > dataframe['kijun_sen_380']) &
            (dataframe['close'] > dataframe['ema440']) &
            (dataframe['tenkan_sen_12'] > dataframe['senkou_a_6']) &
            (dataframe['senkou_a_6'] > dataframe['senkou_b_6'])
        ).astype('int')

        dataframe['trending_over'] = (
            (dataframe['hma67_1h'] > dataframe['ema88']) &
            (dataframe['kijun_sen_20'] > dataframe['close'])
        ).astype('int')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        dataframe = self.slow_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['ichimoku_ok'] > 0) &
                (dataframe['macd_2h'] > dataframe['macdsignal_2h'])
            ), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['trending_over'] > 0
            ), 'sell'] = 1
        return dataframe
