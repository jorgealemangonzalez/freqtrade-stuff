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

def ssl_atr(dataframe, length=7):
    df = dataframe.copy()
    df['smaHigh'] = df['high'].rolling(length).mean() + df['atr']
    df['smaLow'] = df['low'].rolling(length).mean() - df['atr']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1,
                         np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']


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
    # Optimal timeframe for the strategy
    timeframe = '1m'

    # generate signals from the 5m timeframe
    informative_timeframe = '1m'

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
            'kijun_sen_633_3m': {},
            'kijun_sen_20': {},
            'hma888_10m': {},
            'hma800_3m': {},
            'ema440_5m': {},
            'tenkan_sen_20': {},
            'senkou_a_9': {},
            'senkou_b_9': {},
            'hma800_5m': {},
            'ema88_5m': {},
            'kijun_sen_20_5m': {},
            'close': {
                'color': 'black',
            },
        },
        'subplots': {
            'MACD': {
                'macd_12h': {'color': 'blue'},
                'macdsignal_12h': {'color': 'orange'},
            },
        }
    }

    # WARNING setting a stoploss for this strategy doesn't make much sense, as it will buy
    # back into the trend at the next available opportunity, unless the trend has ended,
    # in which case it would sell anyway.

    # Stoploss:
    stoploss = -0.10

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

       # Pares en 3m

        dataframe3m = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="3m")

        create_ichimoku(dataframe3m, conversion_line_period=633,
                        displacement=633, base_line_periods=444, laggin_span=444)

        dataframe3m['hma800'] = ftt.hull_moving_average(dataframe3m, 800)

        dataframe = merge_informative_pair(
            dataframe, dataframe3m, self.timeframe, "3m", ffill=True)

        # Pares en 5m

        dataframe5m = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="5m")

        create_ichimoku(dataframe5m, conversion_line_period=20,
                        displacement=88, base_line_periods=88, laggin_span=88)

        dataframe5m['hma444'] = ftt.hull_moving_average(dataframe5m, 444)
        dataframe5m['hma800'] = ftt.hull_moving_average(dataframe5m, 800)
        dataframe5m['ema440'] = ta.EMA(dataframe5m, timeperiod=440)
        dataframe5m['ema88'] = ta.EMA(dataframe5m, timeperiod=88)

        dataframe = merge_informative_pair(
            dataframe, dataframe5m, self.timeframe, "5m", ffill=True)

        # Pares en 10m

        dataframe10m = resample_to_interval(dataframe, 10)

        dataframe10m['hma888'] = ftt.hull_moving_average(dataframe10m, 888)

        dataframe = merge_informative_pair(
            dataframe, dataframe10m, self.timeframe, "10m", ffill=True)

        # Pares en 12h
        dataframe12h = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="12h")

        # MACD
        macd = ta.MACD(dataframe12h, fastperiod=12,
                       slowperiod=26, signalperiod=9)
        dataframe12h['macd'] = macd['macd']
        dataframe12h['macdsignal'] = macd['macdsignal']

        dataframe = merge_informative_pair(
            dataframe, dataframe12h, self.timeframe, "12h", ffill=True)

        # dataframe normal

        create_ichimoku(dataframe, conversion_line_period=20,
                        displacement=88, base_line_periods=88, laggin_span=88)
        create_ichimoku(dataframe, conversion_line_period=9,
                        displacement=26, base_line_periods=26, laggin_span=52)

        dataframe['ichimoku_ok'] = (
            (dataframe['kijun_sen_633_3m'] > dataframe['hma888_10m']) &
            (dataframe['kijun_sen_633_3m'] > dataframe['hma800_3m']) &
            (dataframe['kijun_sen_20'] > dataframe['kijun_sen_633_3m']) &
            (dataframe['close'] > dataframe['ema440_5m']) &
            (dataframe['tenkan_sen_20'] > dataframe['senkou_a_9']) &
            (dataframe['senkou_a_9'] > dataframe['senkou_b_9'])
        ).astype('int')

        dataframe['trending_over'] = (
            (dataframe['hma800_5m'] > dataframe['ema88_5m']) &
            (dataframe['kijun_sen_20_5m'] > dataframe['close'])
        ).astype('int')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.slow_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                dataframe['ichimoku_ok'] > 0 &
                (dataframe['macd_12h'] > dataframe['macdsignal_12h'])
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['trending_over'] > 0
            ), 'sell'] = 1
        return dataframe
