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
    # La Estrategia es: SymphonIK_v6 (con MACD)
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

    # WARNING setting a stoploss for this strategy doesn't make much sense, as it will buy
    # back into the trend at the next available opportunity, unless the trend has ended,
    # in which case it would sell anyway.

    # Stoploss:
    stoploss = -0.10

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Pares en 1h
        dataframe1h = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1h")

        dataframe1h['hma148'] = ftt.hull_moving_average(dataframe1h, 148)
        dataframe1h['hma67'] = ftt.hull_moving_average(dataframe1h, 67)
        dataframe1h['hma40'] = ftt.hull_moving_average(dataframe1h, 40)

        dataframe1h['Ahma15'] = ftt.hull_moving_average(dataframe1h, 15)
        dataframe1h['Ahma20'] = ftt.hull_moving_average(dataframe1h, 20)
        dataframe1h['Ahma25'] = ftt.hull_moving_average(dataframe1h, 25)
        dataframe1h['Ahma30'] = ftt.hull_moving_average(dataframe1h, 30)
        dataframe1h['Ahma35'] = ftt.hull_moving_average(dataframe1h, 35)
        dataframe1h['Ahma45'] = ftt.hull_moving_average(dataframe1h, 45)
        dataframe1h['Ahma50'] = ftt.hull_moving_average(dataframe1h, 50)
        dataframe1h['Ahma55'] = ftt.hull_moving_average(dataframe1h, 55)
        dataframe1h['Ahma60'] = ftt.hull_moving_average(dataframe1h, 60)

        dataframe1h['Ahma65'] = ftt.hull_moving_average(dataframe1h, 65)
        dataframe1h['Ahma70'] = ftt.hull_moving_average(dataframe1h, 70)
        dataframe1h['Ahma75'] = ftt.hull_moving_average(dataframe1h, 75)
        dataframe1h['Ahma80'] = ftt.hull_moving_average(dataframe1h, 80)
        dataframe1h['Ahma85'] = ftt.hull_moving_average(dataframe1h, 85)
        dataframe1h['Ahma90'] = ftt.hull_moving_average(dataframe1h, 90)
        dataframe1h['Ahma95'] = ftt.hull_moving_average(dataframe1h, 95)
        dataframe1h['Ahma100'] = ftt.hull_moving_average(dataframe1h, 100)
        dataframe1h['Ahma105'] = ftt.hull_moving_average(dataframe1h, 105)
        
        dataframe1h['Ahma110'] = ftt.hull_moving_average(dataframe1h, 110)
        dataframe1h['Ahma115'] = ftt.hull_moving_average(dataframe1h, 115)
        dataframe1h['Ahma120'] = ftt.hull_moving_average(dataframe1h, 120)
        dataframe1h['Ahma125'] = ftt.hull_moving_average(dataframe1h, 125)
        dataframe1h['Ahma130'] = ftt.hull_moving_average(dataframe1h, 130)
        dataframe1h['Ahma135'] = ftt.hull_moving_average(dataframe1h, 135)
        dataframe1h['Ahma140'] = ftt.hull_moving_average(dataframe1h, 140)
        dataframe1h['Ahma145'] = ftt.hull_moving_average(dataframe1h, 145)
        dataframe1h['Ahma150'] = ftt.hull_moving_average(dataframe1h, 150)

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

        # Start Trading

        dataframe['ichimoku_ok'] = (
            (dataframe['macd_1h'] > dataframe['macdsignal_1h']) &
            (dataframe['kijun_sen_380'] > dataframe['hma148_1h']) &
            (dataframe['kijun_sen_380'] > dataframe['hma40_1h']) &
            (dataframe['kijun_sen_12'] > dataframe['kijun_sen_380']) &
            (dataframe['close'] > dataframe['ema440']) &
            (dataframe['tenkan_sen_12'] > dataframe['senkou_b_9']) &
            (dataframe['senkou_a_9'] > dataframe['senkou_b_9'])
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
