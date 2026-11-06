"""陈凯「三周期」共振选股策略（日线选股版）。

策略思想：顺大势、逆小势（陈凯老师的三周期交易法）。
  - 大周期（周K）定性：只在周线上升趋势中做多，震荡/下跌一律放弃
  - 中周期（日K）找三种特征图形：旗形回踩 / 圆弧底右侧 / 收敛突破
  - 小周期切入点（日K模拟）：当日放量、加速起点（20日新高）、回踩转身阳线

选股条件（全部满足）：
  1. 大周期定性：当前周收盘 > 20周均线，且 20周均线向上
  2. 中周期图形：满足「旗形回踩 / 圆弧底右侧 / 收敛突破」三者之一
  3. 小周期切入点：当日放量（>20日均量×1.5）或创20日收盘新高，且收阳线
  4. 流动性过滤：当日成交额 > 1 亿元
  5. 换手率过滤：真实换手率位于 5%~8%（陈凯偏好：短线客活跃区间）

注：换手率使用 baostock 每日批量接口返回的真实换手率（turn 字段，
    存于 stock_daily.turn_rate），非成交额近似。
"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class ThreeCycleStrategy(BaseStrategy):
    """陈凯三周期共振选股策略。

    Attributes:
        webhook_key: 路由到 'three_cycle' 专属飞书机器人。
    """

    webhook_key: str = "three_cycle"

    # 陈凯偏好：换手率 5%~8%（短线客多、适合三周期短线），可按需调整
    TURN_RATE_MIN: float = 5.0
    TURN_RATE_MAX: float = 8.0

    # ── 大周期：周K定性 ──

    @staticmethod
    def _weekly_uptrend(df: pd.DataFrame) -> bool:
        """周线上升趋势：周收盘 > 20周均线，且 20周均线向上。"""
        dt = pd.to_datetime(df["date"])
        weekly_close = df.assign(dt=dt).set_index("dt")["close"].resample("W-FRI").last().dropna()
        if len(weekly_close) < 24:
            return False
        ma20 = weekly_close.rolling(20).mean()
        last_close = weekly_close.iloc[-1]
        last_ma = ma20.iloc[-1]
        prev_ma = ma20.iloc[-4]
        return bool(last_close > last_ma and last_ma > prev_ma)

    # ── 中周期：三种特征图形 ──

    @staticmethod
    def _flag_pattern(df: pd.DataFrame) -> bool:
        """旗形上涨图：两波上涨（高点抬高+低点抬高），当前处于缩量回踩。

        回踩要求：从近20日高点回撤 3%~15%，未跌破前一波低点，且近3日缩量。
        """
        if len(df) < 60:
            return False
        high = df["high"].tail(60)
        low = df["low"].tail(60)
        volume = df["volume"].tail(60)
        close = df["close"].iloc[-1]

        first_high = high.iloc[-40:-20].max()
        second_high = high.iloc[-20:].max()
        first_low = low.iloc[-40:-20].min()
        second_low = low.iloc[-20:].min()

        higher_highs = second_high > first_high          # 高点抬高（第二波更强）
        higher_lows = second_low > first_low             # 低点抬高

        retrace = (second_high - close) / second_high if second_high > 0 else 0.0
        in_pullback = 0.03 <= retrace <= 0.15            # 温和回踩
        above_prev_low = close > first_low               # 未破前低

        vol_pullback = volume.iloc[-3:].mean() < volume.iloc[-13:-3].mean()  # 回踩缩量
        return bool(higher_highs and higher_lows and in_pullback and above_prev_low and vol_pullback)

    @staticmethod
    def _round_bottom_pattern(df: pd.DataFrame) -> bool:
        """圆弧底右侧：近30日先跌后涨（最低点在中段），右侧放量上行。

        圆弧底特点（陈凯）：左侧缩量阴跌，右侧逐渐放量回升。
        """
        if len(df) < 30:
            return False
        window = df.tail(30)
        close = window["close"].reset_index(drop=True)
        volume = window["volume"].reset_index(drop=True)

        min_idx = int(close.idxmin())
        # 最低点应位于窗口的中段（12~24），而非两端
        if not (12 <= min_idx <= 24):
            return False

        left_vol = volume.iloc[:min_idx].mean()
        right_vol = volume.iloc[min_idx:].mean()
        right_volume_expand = right_vol > left_vol * 1.2          # 右侧放量
        right_uptrend = close.iloc[-1] > close.iloc[min_idx] * 1.05  # 右侧已回升
        return bool(right_volume_expand and right_uptrend)

    @staticmethod
    def _convergence_breakout(df: pd.DataFrame) -> bool:
        """收敛突破：近20日高点递减、低点递增（收敛），今日收盘突破收敛上轨。"""
        if len(df) < 25:
            return False
        window = df.tail(25)
        high = window["high"]
        low = window["low"]
        close_today = df["close"].iloc[-1]

        first_half_high = high.iloc[-25:-12].max()
        second_half_high = high.iloc[-12:-1].max()
        first_half_low = low.iloc[-25:-12].min()
        second_half_low = low.iloc[-12:-1].min()

        converging = second_half_high < first_half_high and second_half_low > first_half_low
        breakout = close_today > high.iloc[-20:-1].max()          # 突破近20日高点
        return bool(converging and breakout)

    # ── 小周期：切入点 ──

    @staticmethod
    def _entry_signal(df: pd.DataFrame) -> bool:
        """切入点：当日放量或创20日收盘新高，且收阳线（回踩转身特征）。"""
        if len(df) < 21:
            return False
        last = df.iloc[-1]
        prev_vol_ma = df["volume"].iloc[-21:-1].mean()
        volume_surge = last["volume"] > prev_vol_ma * 1.5
        new_high = last["close"] >= df["close"].iloc[-20:].max()
        yang_line = last["close"] > last["open"]
        return bool((volume_surge or new_high) and yang_line)

    def run(self) -> list[str]:
        """遍历全市场，返回满足三周期共振条件的股票代码列表。"""
        symbols = self.engine.get_local_symbols()
        selected: list[str] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < 60:
                    continue

                # 流动性：当日成交额过亿
                if df["turnover"].iloc[-1] <= 100_000_000:
                    continue

                # 陈凯偏好：真实换手率 5%~8%
                try:
                    turn_rate = float(df["turn_rate"].iloc[-1])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (self.TURN_RATE_MIN <= turn_rate <= self.TURN_RATE_MAX):
                    continue

                # 大周期定性：周线上升趋势
                if not self._weekly_uptrend(df):
                    continue

                # 中周期图形：三选一
                pattern_ok = (
                    self._flag_pattern(df)
                    or self._round_bottom_pattern(df)
                    or self._convergence_breakout(df)
                )
                if not pattern_ok:
                    continue

                # 小周期切入点
                if not self._entry_signal(df):
                    continue

                selected.append(symbol)

            except Exception as exc:
                logger.warning(f"[{symbol}] 三周期策略计算失败：{exc}")
                continue

        logger.info(f"ThreeCycleStrategy 选出 {len(selected)} 只股票")
        return selected
