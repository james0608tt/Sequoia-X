"""即时选股：跳过数据同步，直接对现有数据库跑全部 14 个策略并生成报告。

用法（在 Sequoia-X 目录下）：
    DB_PATH=data/sequoia_v2.server_hfq.bak .venv/bin/python ../quick_select.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.notify.report import ReportGenerator
from sequoia_x.strategy.bowl_rebound import BowlReboundStrategy
from sequoia_x.strategy.FirstNewHigh30DaysBreakoutStrategy import FirstNewHigh30DaysBreakoutStrategy
from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy
from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy
from sequoia_x.strategy.ma_volume import MaVolumeStrategy
from sequoia_x.strategy.ml_surge import MlSurgeStrategy
from sequoia_x.strategy.private_placement import PrivatePlacementStrategy
from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
from sequoia_x.strategy.RPS_YILIN import RpsYilinStrategy
from sequoia_x.strategy.stop_fall import StopFallStrategy
from sequoia_x.strategy.three_cycle import ThreeCycleStrategy
from sequoia_x.strategy.ThreeTimeYilinStrategy_v3 import TripleVolumeBreakoutStrategy
from sequoia_x.strategy.trend_mean_reversion import TrendMeanReversionStrategy
from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy

import main as app  # 复用共振组合定义

settings = get_settings()
engine = DataEngine(settings)

classes = [
    MaVolumeStrategy,
    MlSurgeStrategy,
    TurtleTradeStrategy,
    HighTightFlagStrategy,
    LimitUpShakeoutStrategy,
    UptrendLimitDownStrategy,
    RpsBreakoutStrategy,
    RpsYilinStrategy,
    TripleVolumeBreakoutStrategy,
    BowlReboundStrategy,
    PrivatePlacementStrategy,
    FirstNewHigh30DaysBreakoutStrategy,
    TrendMeanReversionStrategy,
    StopFallStrategy,
    ThreeCycleStrategy,
]

results: dict[str, list[str]] = {}
for cls in classes:
    t0 = time.time()
    sel = cls(engine=engine, settings=settings).run()
    results[cls.__name__] = sel
    print(f"{cls.__name__}: {len(sel)} 只 ({time.time() - t0:.0f}s)", flush=True)

resonance = app._build_resonance_hits(results)
high = sum(h["count"] for h in resonance["high"])
medium = sum(h["count"] for h in resonance["medium"])
print(f"共振：高吸引力 {high} 只，中吸引力 {medium} 只", flush=True)

rpt = ReportGenerator(settings).generate(results, resonance)
print("报告已生成:", rpt)
