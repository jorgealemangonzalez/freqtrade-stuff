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
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
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


class Miku_1m_5m_CSen444v2_N_1_5(IStrategy):
    # La estrategia es: Miku_1m_5m_CEma110v2_N_1_5

    # La estrategia base es: Miku_1m_5m_CSen444v2_N_1_5

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
            'kijun_sen_355_5m': {},
            'tenkan_sen_355_5m': {},
            'senkou_a_100': {},
            'senkou_b_100': {},
            'senkou_a_20': {},
            'senkou_b_20': {},
            'kijun_sen_20': {},
            'tenkan_sen_20': {},
            'tenkan_sen_444': {},
            'senkou_a_9': {},
            'tenkan_sen_9': {},
            'kijun_sen_9': {},
            'ema110_5m': {},

        },
    }

    # WARNING setting a stoploss for this strategy doesn't make much sense, as it will buy
    # back into the trend at the next available opportunity, unless the trend has ended,
    # in which case it would sell anyway.

    # Stoploss:
    stoploss = -0.10

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        if self.dp:
            informative_pairs += [(pair, "5m") for pair in pairs]
        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe="5m")

        create_ichimoku(dataframe5m, conversion_line_period=355, displacement=880, base_line_periods=175, laggin_span=175)

        dataframe5m['ema110'] = ta.EMA(dataframe5m, timeperiod=110)

        dataframe = merge_informative_pair(dataframe, dataframe5m, self.timeframe, "5m", ffill=True)


        create_ichimoku(dataframe, conversion_line_period=20, displacement=88, base_line_periods=88, laggin_span=88)
        create_ichimoku(dataframe, conversion_line_period=9, displacement=26, base_line_periods=26, laggin_span=52)
        create_ichimoku(dataframe, conversion_line_period=444, displacement=444, base_line_periods=444, laggin_span=444)
        create_ichimoku(dataframe, conversion_line_period=100, displacement=88, base_line_periods=440, laggin_span=440)
        create_ichimoku(dataframe, conversion_line_period=40, displacement=88, base_line_periods=176, laggin_span=176)



        dataframe['ichimoku_ok'] = (
                                           (dataframe['kijun_sen_355_5m'] >= dataframe['tenkan_sen_355_5m']) &
                                           (dataframe['senkou_a_100'] > dataframe['senkou_b_100']) &
                                           (dataframe['senkou_a_20'] > dataframe['senkou_b_20']) &
                                           (dataframe['kijun_sen_20'] > dataframe['tenkan_sen_444']) &
                                           (dataframe['senkou_a_9'] > dataframe['senkou_a_20']) &
                                           (dataframe['tenkan_sen_20'] >= dataframe['kijun_sen_20']) &
                                           (dataframe['tenkan_sen_9'] >= dataframe['tenkan_sen_20']) &
                                           (dataframe['tenkan_sen_9'] >= dataframe['kijun_sen_9'])
                                   ).astype('int') * 4

        dataframe['trending_over'] = (
                                         (dataframe['ema110_5m'] > dataframe['close'])
                                     ).astype('int') * 1

        return dataframe

    def fast_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # none atm
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Obelisk_Ichimoku_Slow does not use trailing stop or roi and should be safe to backtest at 5m
        # if self.config['runmode'].value in ('backtest', 'hyperopt'):
        #     assert (timeframe_to_minutes(self.timeframe) <= 5), "Backtest this strategy in 5m or 1m timeframe."

        if self.timeframe == self.informative_timeframe:
            dataframe = self.slow_tf_indicators(dataframe, metadata)
        else:
            assert self.dp, "DataProvider is required for multiple timeframes."

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
            informative = self.slow_tf_indicators(informative.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                               ffill=True)
            # don't overwrite the base dataframe's OHLCV information
            skip_columns = [(s + "_" + self.informative_timeframe) for s in
                            ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (
                not s in skip_columns) else s, inplace=True)

        dataframe = self.fast_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['ichimoku_ok'] > 0)
            & (dataframe['trending_over'] <= 0)
            , 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['trending_over'] > 0)
            , 'sell'] = 1
        return dataframe
