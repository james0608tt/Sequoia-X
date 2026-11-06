"""ML 选股策略：预测「未来 20 个交易日内最高涨幅 ≥15%」的概率。

模型：HistGradientBoosting（sklearn），21 个技术特征，严格按时间切分训练
（训练 <2025-06-01，验证 2025-06~2026-01，测试 ≥2026-01）。

经止损回测验证的最优执行规则（2026-01~2026-09 测试期，非ST版模型）：
- 信号：当日预测概率 ≥ 0.7（日均约 189 只）
- 执行：T+1 开盘买入，持有满 20 个交易日，不带止损/止盈
- 回测：单笔均值 +3.28% / 中位 +1.90% / 胜率 59.4%；等权组合 9 个月 +8.0%，最大回撤 -19.6%
- 注意：止损(-10%)+止盈(+15%)反而把期望打到约 0（止损会砍掉先回撤后反弹的赢家）

模型文件：data/surge20_model_nost.joblib（由 scripts/train_nost_backtest.py 生成）。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)

FEATURES = [
    "ret_1", "ret_5", "ret_20", "ret_60",
    "vol_ratio_5", "vol_ratio_20",
    "dist_ma20", "dist_ma60", "ma20_above_ma60",
    "dist_high20", "dist_high60", "drawdown_60",
    "atr_pct", "amplitude_10", "up_days_10",
    "days_above_ma20_10",
    "rps_20", "rps_60", "rps_120",
    "turn_rate", "log_turnover",
]


class MlSurgeStrategy(BaseStrategy):
    """ML 20日大涨概率策略。

    加载训练好的梯度提升模型，对最新交易日全市场打分，
    返回概率 ≥ 阈值的 Top N 股票（按概率降序）。

    Attributes:
        webhook_key: 路由到 'ml_surge' 专属飞书机器人。
    """

    webhook_key: str = "ml_surge"

    MIN_PROB: float = 0.7
    MAX_RESULTS: int = 50
    HORIZON: int = 20
    TARGET_RET: float = 0.15

    def _build_latest_features(self) -> pd.DataFrame | None:
        """为全市场最新交易日构建特征（复用训练管线逻辑）。"""
        import sqlite3

        with sqlite3.connect(self.engine.db_path) as conn:
            df = pd.read_sql(
                "SELECT symbol, date, open, high, low, close, volume, turnover, turn_rate"
                " FROM stock_daily",
                conn,
            )
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume", "turnover", "turn_rate"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        g = df.groupby("symbol", group_keys=False)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]
        ma20 = g["close"].transform(lambda s: s.rolling(20).mean())
        ma60 = g["close"].transform(lambda s: s.rolling(60).mean())
        df["ret_1"] = close / g["close"].shift(1) - 1
        df["ret_5"] = close / g["close"].shift(5) - 1
        df["ret_20"] = close / g["close"].shift(20) - 1
        df["ret_60"] = close / g["close"].shift(60) - 1
        df["vol_ratio_5"] = vol / g["volume"].transform(lambda s: s.rolling(5).mean())
        df["vol_ratio_20"] = vol / g["volume"].transform(lambda s: s.rolling(20).mean())
        df["dist_ma20"] = close / ma20 - 1
        df["dist_ma60"] = close / ma60 - 1
        df["ma20_above_ma60"] = (ma20 > ma60).astype(float)
        df["dist_high20"] = close / g["high"].transform(lambda s: s.rolling(20).max()) - 1
        df["dist_high60"] = close / g["high"].transform(lambda s: s.rolling(60).max()) - 1
        df["drawdown_60"] = close / g["high"].transform(lambda s: s.rolling(60).max()) - 1
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr14 = tr.groupby(df["symbol"], group_keys=False).transform(lambda s: s.rolling(14).mean())
        df["atr_pct"] = atr14 / close
        amp = (high - low) / close
        df["amplitude_10"] = amp.groupby(df["symbol"], group_keys=False).transform(lambda s: s.rolling(10).mean())
        df["up_days_10"] = (df["ret_1"] > 0).astype(float).groupby(df["symbol"], group_keys=False).transform(lambda s: s.rolling(10).mean())
        above = (close > ma20).astype(float)
        df["days_above_ma20_10"] = above.groupby(df["symbol"], group_keys=False).transform(lambda s: s.rolling(10).mean())
        df["turn_rate"] = df["turn_rate"].fillna(0.0)
        df["log_turnover"] = np.log10(df["turnover"].clip(lower=1e6))
        for p in (20, 60, 120):
            r = close / g["close"].shift(p) - 1
            df[f"rps_{p}"] = r.groupby(df["date"]).rank(pct=True) * 100

        latest = df.dropna(subset=FEATURES)
        latest = latest[latest["date"] == latest["date"].max()]
        return latest

    def run(self) -> list[str]:
        """加载模型并对最新交易日全市场打分，返回高概率股票代码（剔除 ST）。"""
        from joblib import load

        model_path = Path(self.engine.db_path).parent / "surge20_model_nost.joblib"
        if not model_path.exists():
            logger.warning(f"ML 模型不存在：{model_path}（先运行 scripts/train_nost_backtest.py）")
            return []

        model = load(model_path)
        latest = self._build_latest_features()
        if latest is None or latest.empty:
            logger.warning("ML 策略无可用特征数据")
            return []

        # 剔除最新交易日仍为 ST 的股票（与训练口径一致）
        latest = latest.copy()
        latest["prob"] = model.predict_proba(latest[FEATURES])[:, 1]
        latest = self._exclude_current_st(latest)
        hits = latest[latest["prob"] >= self.MIN_PROB]
        hits = hits.sort_values("prob", ascending=False).head(self.MAX_RESULTS)

        logger.info(
            f"MlSurgeStrategy 选出 {len(hits)} 只股票 "
            f"（阈值 {self.MIN_PROB:.0%}，最新日 {latest['date'].max().date()}）"
        )
        return hits["symbol"].tolist()

    def _exclude_current_st(self, latest: pd.DataFrame) -> pd.DataFrame:
        """剔除当前仍为 ST 的股票（用 st_flag 最新可得日期，ST 状态变化慢）。"""
        import sqlite3

        with sqlite3.connect(self.engine.db_path) as conn:
            st_syms = {
                r[0] for r in conn.execute(
                    "SELECT symbol FROM st_flag WHERE isST=1"
                    " AND date = (SELECT MAX(date) FROM st_flag)"
                ).fetchall()
            }
        if st_syms:
            latest = latest[~latest["symbol"].isin(st_syms)]
        return latest
