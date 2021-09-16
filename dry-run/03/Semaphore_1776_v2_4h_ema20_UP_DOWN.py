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


class Semaphore_1776_v2_4h_ema20_UP_DOWN(IStrategy):
    # Semaphore_1776_v2_4h_ema20_UP_DOWN
    # Optimal timeframe for the strategy
    timeframe = '5m'

    # generate signals from the 5m timeframe
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
    stoploss = -0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe)
                             for pair in pairs]
        if self.dp:
            informative_pairs += [(pair, "1h") for pair in pairs]
            informative_pairs += [("BTC/USDT", "4h")]
        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Pares en 5m

        dataframe5m = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="5m")
            # Ichimoku 380_5m equivale al 633_3m
        create_ichimoku(dataframe5m, conversion_line_period=380,
                        displacement=633, base_line_periods=380, laggin_span=266)
            # Ickimoku 12_5m equivale al 20_3m
        create_ichimoku(dataframe5m, conversion_line_period=12,
                        displacement=88, base_line_periods=53, laggin_span=53)
        create_ichimoku(dataframe5m, conversion_line_period=20,
                        displacement=88, base_line_periods=88, laggin_span=88)
        create_ichimoku(dataframe5m, conversion_line_period=178,
                        displacement=880, base_line_periods=88, laggin_span=88)
        create_ichimoku(dataframe5m, conversion_line_period=355,
                        displacement=880, base_line_periods=175, laggin_span=175)
        create_ichimoku(dataframe5m, conversion_line_period=1776,
                        displacement=880, base_line_periods=880, laggin_span=880)

            # Hma 480_5m equivale a la hma800_3m
        dataframe5m['hma480'] = ftt.hull_moving_average(dataframe5m, 480)
        dataframe5m['hma800'] = ftt.hull_moving_average(dataframe5m, 800)
        dataframe5m['ema440'] = ta.EMA(dataframe5m, timeperiod=440)
        dataframe5m['ema88'] = ta.EMA(dataframe5m, timeperiod=88)

        dataframe = merge_informative_pair(
            dataframe, dataframe5m, self.timeframe, "5m", ffill=True)

        # Pares en 1h
        dataframe1h = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe="1h")

        dataframe1h['hma148'] = ftt.hull_moving_average(dataframe1h, 148)
        dataframe1h['hma67'] = ftt.hull_moving_average(dataframe1h, 67)
        dataframe1h['hma40'] = ftt.hull_moving_average(dataframe1h, 40)
        dataframe1h['hma100'] = ftt.hull_moving_average(dataframe1h, 100)
        dataframe1h['hma120'] = ftt.hull_moving_average(dataframe1h, 120)
        dataframe1h['hma80'] = ftt.hull_moving_average(dataframe1h, 80)
        dataframe1h['hma75'] = ftt.hull_moving_average(dataframe1h, 75)
        dataframe1h['hma85'] = ftt.hull_moving_average(dataframe1h, 85)
        dataframe1h['hma90'] = ftt.hull_moving_average(dataframe1h, 90)
        dataframe1h['hma95'] = ftt.hull_moving_average(dataframe1h, 95)
        dataframe1h['hma105'] = ftt.hull_moving_average(dataframe1h, 105)
        dataframe1h['hma110'] = ftt.hull_moving_average(dataframe1h, 110)
        dataframe1h['hma115'] = ftt.hull_moving_average(dataframe1h, 115)
        dataframe1h['hma125'] = ftt.hull_moving_average(dataframe1h, 125)
        dataframe1h['hma130'] = ftt.hull_moving_average(dataframe1h, 130)
        dataframe1h['hma108'] = ftt.hull_moving_average(dataframe1h, 108)
        dataframe1h['hma107'] = ftt.hull_moving_average(dataframe1h, 107)
        dataframe1h['hma106'] = ftt.hull_moving_average(dataframe1h, 106)
        dataframe1h['hma127'] = ftt.hull_moving_average(dataframe1h, 127)

        dataframe = merge_informative_pair(
            dataframe, dataframe1h, self.timeframe, "1h", ffill=True)

        # BTC/USDT 4h

        dataframe4h = self.dp.get_pair_dataframe(
            pair="BTC/USDT", timeframe="4h")

        dataframe4h['ema20'] = ta.EMA(dataframe4h, timeperiod=20)

        dataframe = merge_informative_pair(
            dataframe, dataframe4h, self.timeframe, "4h", ffill=True)

        # dataframe normal

        create_ichimoku(dataframe, conversion_line_period=20,
                        displacement=88, base_line_periods=88, laggin_span=88)
        create_ichimoku(dataframe, conversion_line_period=380,
                        displacement=633, base_line_periods=380, laggin_span=266)
        create_ichimoku(dataframe, conversion_line_period=12,
                        displacement=88, base_line_periods=53, laggin_span=53)
        create_ichimoku(dataframe, conversion_line_period=9,
                        displacement=26, base_line_periods=26, laggin_span=52)
        create_ichimoku(dataframe, conversion_line_period=444,
                        displacement=444, base_line_periods=444, laggin_span=444)
        create_ichimoku(dataframe, conversion_line_period=100,
                        displacement=88, base_line_periods=440, laggin_span=440)
        create_ichimoku(dataframe, conversion_line_period=40,
                        displacement=88, base_line_periods=176, laggin_span=176)

        dataframe['hma480'] = ftt.hull_moving_average(dataframe, 480)
        dataframe['hma800'] = ftt.hull_moving_average(dataframe, 800)
        dataframe['ema440'] = ta.EMA(dataframe, timeperiod=440)
        dataframe['ema88'] = ta.EMA(dataframe, timeperiod=88)

        # Start Trading

        dataframe['ichimoku_ok'] = (
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
