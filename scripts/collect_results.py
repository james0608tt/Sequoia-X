"""收集 Sequoia-X 七个策略的选股结果并保存为文本文件。"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy
from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy
from sequoia_x.strategy.ma_volume import MaVolumeStrategy
from sequoia_x.strategy.private_placement import PrivatePlacementStrategy
from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy

settings = get_settings()
engine = DataEngine(settings)

strategies = [
    MaVolumeStrategy(engine=engine, settings=settings),
    TurtleTradeStrategy(engine=engine, settings=settings),
    HighTightFlagStrategy(engine=engine, settings=settings),
    LimitUpShakeoutStrategy(engine=engine, settings=settings),
    UptrendLimitDownStrategy(engine=engine, settings=settings),
    RpsBreakoutStrategy(engine=engine, settings=settings),
    PrivatePlacementStrategy(engine=engine, settings=settings),
]

results = {}
for s in strategies:
    name = type(s).__name__
    sel = s.run()
    results[name] = sel
    print(f"{name}: {len(sel)} 只")

# 股票名称映射（akshare 免费列表，无 token）
import akshare as ak
name_df = ak.stock_info_a_code_name()
name_map = dict(zip(name_df["code"].astype(str).str.zfill(6), name_df["name"]))

today = date.today().strftime("%Y-%m-%d")
lines = [f"Sequoia-X 选股结果（{today}）", "=" * 30, ""]
for name, sel in results.items():
    lines.append(f"【{name}】共 {len(sel)} 只")
    if sel:
        for code in sel:
            lines.append(f"  {code}  {name_map.get(code, '')}")
    else:
        lines.append("  （无）")
    lines.append("")

text = "\n".join(lines)
with open("/Users/taofeng/Documents/deepseek-harness/default-workspace/选股结果.txt", "w", encoding="utf-8") as f:
    f.write(text)
with open("/Users/taofeng/Documents/deepseek-harness/default-workspace/选股结果.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("已保存 选股结果.txt / 选股结果.json")
