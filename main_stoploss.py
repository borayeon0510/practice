# @title Default title text
from decimal import *
import sys
import pytz
import os
import logging
import datetime
import time
import pandas as pd
import traceback
import numpy as np
import simplejson
import requests
import math
from binance_api import BinanceApi
from gy_telegram.gy_telegram import TelegramGroup


telegram = TelegramGroup(TelegramGroup.token)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logging.getLogger('requests').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
project_path = os.path.dirname(__file__)


class coin_trade:
    def __init__(self, st_name):
        self.st_name = st_name
        config = simplejson.loads(open(os.path.join(project_path, f'{st_name}.json')).read())
        logger.info("load setting from a json file...")
        logger.info(config)
        # telegram.send_telegram_msg(chat_id=self.telegram_channel, text=f"run {config['name']}")

        self.telegram_channel = config["telegram_channel"]
        self.starting_date = config["setting"]['starting_date']
        self.asset_allocation_type = config["setting"]['asset_allocation_type']  ## 전략1 = FIX_RATION, 전략2 = REBALANCE_RATION
        self.starting_capital = config["setting"]['starting_capital']
        self.leverage = config["setting"]['leverage']
        self.strategy_ratio = config["setting"]['strategy_ratio']
        self.signal_get_period = config["setting"]['signal_get_period']
        self.fee = config["setting"]['fee']
        self.profit = config["setting"]['profit']
        config = simplejson.loads(open(os.path.join(project_path, f'{st_name}.json')).read())
        self.api = BinanceApi(config["binance"]['api_key'], config["binance"]['secret'])

    def get_future_candle_history(self, order_currency, base_currency='USDT', interval='1d'):
        logger.info("get_future_candle_history")
        if interval not in ['1m', '3m', '5m', '10m', '30m', '1h', '6h', '12h', '1d']:
            raise Exception(f"{interval} is not support.")

        res = requests.get(
            f'https://www.binance.com/fapi/v1/klines?symbol={order_currency}{base_currency}&interval={interval}')
        res = res.json()
        df = None
        if len(res) > 0:
            df = pd.DataFrame(res)
            df.rename(columns={0: 'date', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
            df['date'] = df['date'].apply(lambda x: pd.Timestamp(int(x*1000000)))
            df['open'] = df['open'].apply(lambda x: float(x))
            df['high'] = df['high'].apply(lambda x: float(x))
            df['low'] = df['low'].apply(lambda x: float(x))
            df['close'] = df['close'].apply(lambda x: float(x))
            df['volume'] = df['volume'].apply(lambda x: float(x))
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        # utc 0시 0분 0초에 호출하면 당일 캔들은 아직 생성되지 않음. 만약 약간 시간이 지나면 date확인하고 날려주는게 좋을것으로 보임
        # df = df.drop(df.tail(1).index)
        df = df.sort_values(by=['date'], ascending=True)
        df.set_index('date', inplace=True)
        return df

    def signal_final(self, order_currency):
        today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        while True:
            df2 = self.get_future_candle_history(order_currency)
            if df2.index[-1].strftime('%Y-%m-%d') == today:
                break
            time.sleep(5)

        self.today = df2.index[-1].strftime('%Y-%m-%d')
        self.yesterday = df2.index[-2]
        df2['momentum'] = df2['open'] + 0.8 * (df2['high'].shift(1) - df2['low'].shift(1))
        df2['skew'] = df2['close'].shift(1).pct_change().rolling(10).skew()
        df2['chg'] = df2['close'].pct_change()
        df2['rtn_momentum'] = np.where(df2['momentum'] <= df2['high'],
                                       df2['close'] / df2['momentum'] - 1 - self.fee * 2, 0) * self.leverage
        df2['rtn_skew_1'] = np.where(df2['skew'] <= -1, -(df2['close'] / df2['close'].shift(1) - 1) - self.fee * 2,
                                     0) * self.leverage
        df2['rtn_skew_2'] = np.where((df2['skew'].shift(1) <= -1) & (-df2['chg'].shift(1) < 0),
                                     -(df2['close'] / df2['close'].shift(1) - 1), 0) * self.leverage

        df = df2.loc[(datetime.datetime.strptime(self.starting_date, '%Y-%m-%d') - datetime.timedelta(days=1)):, :].copy()

        for i in df.index:
            if i == df.index[0]:
                df.loc[i, 'capital_momentum'] = self.starting_capital * self.strategy_ratio['momentum']
                df.loc[i, 'capital_skew'] = self.starting_capital * self.strategy_ratio['skew']
                df.loc[i, 'capital_total'] = df.loc[i, 'capital_momentum'] + df.loc[i, 'capital_skew']
                df.loc[i, 'pnl'] = df.loc[i, 'capital_total'] - self.starting_capital

                df.loc[i, 'q_momentum'] = 0
                df.loc[i, 'q_skew'] = 0

            else:
                if self.asset_allocation_type == "FIX_RATION":
                    df.loc[i, 'capital_momentum'] = df.shift(1).loc[i, 'capital_total'] * self.strategy_ratio[
                        'momentum'] * (1 + df.loc[i, 'rtn_momentum'])
                    df.loc[i, 'capital_skew'] = df.shift(1).loc[i, 'capital_total'] * self.strategy_ratio['skew'] * (
                                1 + df.loc[i, 'rtn_skew_1']) \
                                                + np.nan_to_num(df.shift(2).loc[i, 'capital_total']) * \
                                                self.strategy_ratio['skew'] * df.loc[i, 'rtn_skew_2']
                    df.loc[i, 'capital_total'] = df.loc[i, 'capital_momentum'] + df.loc[i, 'capital_skew']
                    df.loc[i, 'pnl'] = df.loc[i, 'capital_total'] - df.shift(1).loc[i, 'capital_total']

                elif self.asset_allocation_type == "REBALANCE_RATION":
                    df.loc[i, 'capital_momentum'] = df.shift(1).loc[i, 'capital_total'] * self.strategy_ratio[
                        'momentum'] * (1 + df.loc[i, 'rtn_momentum'])
                    df.loc[i, 'capital_skew'] = df.shift(1).loc[i, 'capital_total'] * self.strategy_ratio['skew'] * (
                                1 + df.loc[i, 'rtn_skew_1']) \
                                                + (np.nan_to_num(df.shift(2).loc[i, 'capital_skew']) + np.nan_to_num(
                        df.shift(2).loc[i, 'capital_momentum'])) * self.strategy_ratio['skew'] * df.loc[i, 'rtn_skew_2']
                    df.loc[i, 'capital_total'] = df.loc[i, 'capital_momentum'] + df.loc[i, 'capital_skew']
                    df.loc[i, 'pnl'] = df.loc[i, 'capital_total'] - df.shift(1).loc[i, 'capital_total']

                    ##리밸런싱 룰
                    if df.loc[i, 'capital_total'] >= self.starting_capital * (
                            1 + self.profit):  ##수익이 self.profit 도달시 리밸런싱
                        df.loc[i, 'capital_total'] = self.starting_capital

                    if (i + datetime.timedelta(1)).strftime('%d') == '01':  ##다음날이 다음달의 1일이면 리밸런싱
                        df.loc[i, 'capital_total'] = self.starting_capital

                if df.loc[i, 'rtn_momentum'] != 0:
                    df.loc[i, 'q_momentum'] = self.round_down(
                        self.leverage * df.shift(1).loc[i, 'capital_total'] * self.strategy_ratio['momentum'] / df.loc[
                            i, 'momentum'], 3)
                else:
                    df.loc[i, 'q_momentum'] = 0
                if df.loc[i, 'rtn_skew_1'] != 0:
                    df.loc[i, 'q_skew'] = self.round_down(
                        self.leverage * df.shift(1).loc[i, 'capital_total'] * self.strategy_ratio['skew'] /
                        df.shift(1).loc[i, 'close'], 3)
                else:
                    df.loc[i, 'q_skew'] = 0

        if self.today == self.starting_date:
            self.NAV = self.starting_capital
            self.exec_hist = {"momentum": {"d-1": 0}, "skew": {"d-1": 0, "d-2": 0}}

        else:
            self.NAV = df.loc[self.yesterday, 'capital_total']
            self.exec_hist = {"momentum": {"d-1": df.loc[self.yesterday, 'q_momentum']},
                              "skew": {"d-1": df.loc[self.yesterday, 'q_skew'], \
                                       "d-2": df.shift(1).loc[self.yesterday, 'q_skew']}}

        self.df = df
        return df

    def signal_calculation_final(self, order_currency):
        ##momentum (일중 실시간 업데이트)
        today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        while True:
            df = self.get_future_candle_history(order_currency)
            if df.index[-1].strftime('%Y-%m-%d') == today:
                break
            time.sleep(5)

        high = df.tail(1).high.values[0]
        df = self.df
        high = float(high)

        if df['momentum'].iloc[-1] <= high:
            momentum = 1
        else:
            momentum = 0

        ##skew
        if df['skew'].iloc[-1] <= -1:
            skew = -1
        else:
            skew = 0

        signal_calculation_final = {"momentum": momentum, "skew": skew}
        logger.info(f"{high}\t?\t{df['momentum'].iloc[-1]}\t{signal_calculation_final}")
        return signal_calculation_final

    def liquidate_calculation_final(self, i):
        ##momentum
        df = self.df.copy()
        if df['momentum'].iloc[-2] <= df['high'].iloc[-2]:
            momentum = -1
        else:
            momentum = 0

        ##skew
        if df['skew'].iloc[-2] <= -1 and -(df['chg'].iloc[-2]) > 0:
            skew_1 = 1
        else:
            skew_1 = 0
        try:
            if df['skew'].iloc[-3] <= -1 and -(df['chg'].iloc[-3]) < 0:
                skew_2 = 1
            else:
                skew_2 = 0
        except Exception:
            skew_2 = 0
        liquidate_calculation_final = {"momentum": momentum, "skew_1": skew_1, "skew_2": skew_2}

        return liquidate_calculation_final

    def order_execution(self, ticker):
        telegram.send_telegram_msg(chat_id=coin_trade.telegram_channel, text=f"order_execution")
        ##UTC 00 00 00 01에 업데이트 시작
        price = {"momentum": self.df['momentum'].iloc[-1], "skew": self.df['close'].iloc[-2]}

        ## init의 self.asset_allocation_type에서 선택한 전략1,2에 따라 수량 정해짐
        order_amount = {"momentum": self.round_down(
            self.leverage * self.NAV * self.signal_calculation_final(ticker)['momentum'] * self.strategy_ratio[
                'momentum'] / price['momentum'], 3), \
                        "skew": self.round_down(
                            self.leverage * self.NAV * 
                            self.strategy_ratio['skew'] / price['skew'], 3)*self.signal_calculation_final(ticker ['skew'] }

        total_qty = 0
        _text = ''
        if morning_count == 0:
            order_amount_liquidate = {
                "momentum": self.exec_hist['momentum']['d-1'] * self.liquidate_calculation_final(ticker)['momentum'], \
                "skew_1": self.exec_hist['skew']['d-1'] * self.liquidate_calculation_final(ticker)['skew_1'], \
                "skew_2": self.exec_hist['skew']['d-2'] * self.liquidate_calculation_final(ticker)['skew_2']}

            ##UTC 00 00 00 01 에 한번 진행 (신규 진입, 오버나잇  포지션 청산)
            ##주문시 self.leverage 반영해야함
            # TODO: 수량 합쳐서 한번에 주문내기 로그는 따로 남기기
            _text = _text + f"logging order: {ticker} skew {order_amount['skew']}\n"
            _text = _text + f"logging order: {ticker} skew_1 {order_amount_liquidate['skew_1']}\n"
            _text = _text + f"logging order: {ticker} skew_2 {order_amount_liquidate['skew_2']}\n"
            _text = _text + f"logging order: {ticker} momentum {order_amount_liquidate['momentum']}\n"

            total_qty += order_amount['skew']
            total_qty += order_amount_liquidate['skew_1']
            total_qty += order_amount_liquidate['skew_2']
            total_qty += order_amount_liquidate['momentum']
            # self.place_order(ticker, order_amount['skew'], '시장가')
            # self.place_order(ticker, order_amount_liquidate['skew_1'], '시장가')
            # self.place_order(ticker, order_amount_liquidate['skew_2'], '시장가')
            # self.place_order(ticker, order_amount_liquidate['momentum'], '시장가')
            ##UTC 00 00 00 01 이후 실시간 진행 (시그널 잡힐 시 신규 진입)
            ##주문시 self.leverage 반영해야함
        total_qty += order_amount['momentum']
        _text = _text + f"logging order: momentum {ticker} {order_amount['momentum']}"
        # self.place_order(ticker, order_amount['momentum'], '시장가')
        logger.info(_text)
        telegram.send_telegram_msg(chat_id=self.telegram_channel, text=_text)
        self.place_order(ticker, total_qty)

    def round_up(self, n, decimals=0):
        multiplier = 10 ** decimals
        return math.ceil(n * multiplier) / multiplier

    def round_down(self, n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n * multiplier) / multiplier

    def place_order(self, ticker, amount):
        ticker = ticker.upper()
        qty = self.api.adjust_future_qty(order_currency=ticker, base_currency='USDT', qty=Decimal(str(abs(amount))))
        side = 'bid' if amount > 0 else 'ask'
        _text = f"place order future: {ticker} {amount} -(adj)> {side} {qty}"
        _text = f"place order future: {ticker} {side} {qty}"
        logger.info(_text)
        telegram.send_telegram_msg(chat_id=self.telegram_channel, text=_text)
        if qty > 0:
            self.api.place_order_future(qty=qty, side=side, order_type='MARKET', order_currency=ticker, base_currency='USDT')
            # TODO
            with open(f'{self.st_name}_data.json', 'w') as f:
                data = {'positions': {}}
                simplejson.dumps(data, f)

    def print_positions(self):
        positions = coin_trade.api.get_future_position()
        logger.info(positions)



##UTC 00 00 00 01에 업데이트 시작 (self.NAV, 가격 데이터 크롤링, 시그널 계산 등)
##skew sell 전략의 경우 UTC 00 00 00 01에 시그널 생성, 청산, 신규진입 한번에 진행
##momentum buy 전략의 경우 UTC 00 00 00 01에 시그널 생성, 청산 진행 -> 신규진입은 실시간 진행 (self.signal_calculation_final 실시간 업데이트 필요)
morning_count, intra_count, data_count = 1, 0, 0
env = sys.argv[1]
# env = 'coin_trade_1'
coin_trade = coin_trade(env)
while True:
    try:
        _now = pd.Timestamp(datetime.datetime.utcnow().replace(microsecond=0))
        # 매일 utc 0시에 한번 호출
        if _now.hour == 0 and _now.minute == 0 and _now.second == 0:
            morning_count, intra_count, data_count = 0, 0, 0
            logger.info(f"{_now} start")
            df_balance = coin_trade.api.get_future_balance()
            telegram.send_telegram_msg(chat_id=coin_trade.telegram_channel, text=f"{_now} start. balance USDT: {df_balance['balance'].get('USDT', 0)}")
            coin_trade.signal_final('BTC')
            coin_trade.order_execution('BTC')
            morning_count += 1
            data_count += 1
            time.sleep(5)  # 로직 2번 안타게 타임 슬립

        if data_count == 0:
            coin_trade.signal_final('BTC')
            data_count += 1

        # 장중 실시간 매수 체결 로직
        if intra_count == 0 and _now.hour <= 23 and _now.minute < 59:
            if coin_trade.signal_calculation_final('BTC')['momentum'] == 1:
                intra_count += 1
                coin_trade.order_execution('BTC')
            time.sleep(5)  # 부하 우회

    except Exception as e:
        # logger.error(e)
        traceback.print_exc()
    time.sleep(0.01)  # 부하 우회