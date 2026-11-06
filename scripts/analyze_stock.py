"""单只股票全策略体检：002606 大连电瓷。

用法（在 Sequoia-X 目录）：
    .venv/bin/python ../analyze_stock.py 002606
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import get_settings
from sequoia_x.data.engine import DataEngine

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "002606"

settings = get_settings()
engine = DataEngine(settings)
df = engine.get_ohlcv(SYMBOL)
if df.empty:
    print(f"数据库中没有 {SYMBOL} 的数据")
    sys.exit(1)

df = df.sort_values("date").reset_index(drop=True)
for c in ["open", "high", "low", "close", "volume", "turnover", "turn_rate"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

last = df.iloc[-1]
prev = df.iloc[-2]

print(f"股票: {SYMBOL}  最新数据日期: {last['date']}  总K线数: {len(df)}")
print(f"最新收盘: {last['close']:.2f} (后复权)  开:{last['open']:.2f} 高:{last['high']:.2f} 低:{last['low']:.2f}")
print(f"成交额: {last['turnover']/1e8:.2f} 亿  换手率: {last['turn_rate']:.2f}%  昨收: {prev['close']:.2f}")
print(f"当日涨跌: {last['close']/prev['close']*100-100:+.2f}%  实体: {'阳' if last['close']>last['open'] else '阴'}线")
print()

# ── 指标面板 ──
close = df["close"]
ma = {n: close.rolling(n).mean().iloc[-1] for n in (5, 10, 14, 20, 28, 57, 60, 114)}
print("── 均线 ──")
for n, v in ma.items():
    pos = "上方" if last["close"] > v else "下方"
    print(f"  MA{n:<4}: {v:>10.2f}  价格在{pos}")

ema10 = close.ewm(span=10, adjust=False).mean()
short_trend = ema10.ewm(span=10, adjust=False).mean().iloc[-1]
bull_bear = np.mean([ma[14], ma[28], ma[57], ma[114]])
print(f"  短期趋势线(EMA10二次平滑): {short_trend:.2f}  多空线(14/28/57/114均值): {bull_bear:.2f}  趋势线在{'上' if short_trend > bull_bear else '下'}")

low9 = df["low"].rolling(9).min()
high9 = df["high"].rolling(9).max()
rsv = ((close - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
d = k.ewm(alpha=1 / 3, adjust=False).mean()
j = 3 * k - 2 * d
print(f"  KDJ(9,3,3): K={k.iloc[-1]:.1f} D={d.iloc[-1]:.1f} J={j.iloc[-1]:.1f}")

vol = df["volume"]
vol_ma20 = vol.rolling(20).mean().iloc[-1]
vol_ma5 = vol.rolling(5).mean().iloc[-1]
print(f"  量比(今/20日均): {last['volume']/vol_ma20:.2f}  5日均量: {vol_ma5/1e4:.0f}万手")

pct_60 = close.iloc[-1] / close.iloc[-61] - 1
pct_120 = close.iloc[-1] / close.iloc[-121] - 1
pct_5 = close.iloc[-1] / close.iloc[-6] - 1
high60 = df["high"].iloc[-60:].max()
low60 = df["low"].iloc[-60:].min()
print(f"  60日涨幅: {pct_60*100:+.1f}%  120日涨幅: {pct_120*100:+.1f}%  近5日: {pct_5*100:+.1f}%")
print(f"  60日最高: {high60:.2f}  60日最低: {low60:.2f}  60日回撤(从高点): {(last['close']/high60-1)*100:+.1f}%")

tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
atr14 = tr.rolling(14).mean().iloc[-1]
print(f"  ATR14: {atr14:.3f}")

# ── RPS 排名（全市场）──
with sqlite3.connect(engine.db_path) as conn:
    mkt = pd.read_sql("SELECT symbol, date, close FROM stock_daily", conn)
mkt["date"] = pd.to_datetime(mkt["date"])
mkt = mkt.sort_values(["symbol", "date"])
mkt["c120"] = mkt.groupby("symbol")["close"].shift(120)
mkt["ret120"] = (mkt["close"] - mkt["c120"]) / mkt["c120"]
latest = mkt[mkt["date"] == mkt["date"].max()].dropna(subset=["ret120"])
latest["rps"] = latest["ret120"].rank(pct=True) * 100
rps120 = latest.loc[latest["symbol"] == SYMBOL, "rps"]
rps120 = float(rps120.iloc[0]) if len(rps120) else float("nan")
print(f"  RPS120 全市场排名: {rps120:.1f} (前{(100-rps120):.1f}%)")
print()

# ── 逐策略判定 ──
print("── 14 策略逐项判定（数据截至 " + str(last["date"]) + "）──")

def verdict(name, conds):
    ok = all(conds.values())
    print(f"  {name}: {'✅ 符合' if ok else '❌ 不符合'}")
    for k, v in conds.items():
        print(f"      {k}: {'✓' if v else '✗'}")

# 1 海龟
h20 = df["high"].shift(1).rolling(20).max().iloc[-1]
verdict("海龟交易 TurtleTrade", {
    f"突破前20日高点(前高{h20:.2f} vs 收{last['close']:.2f})": last["close"] > h20,
    "成交额过亿": last["turnover"] > 1e8,
    "实体阳线": last["close"] > last["open"],
    "真涨(收>昨收)": last["close"] > prev["close"],
})

# 2 均量
ma5_prev, ma20_prev = close.rolling(5).mean().iloc[-2], close.rolling(20).mean().iloc[-2]
ma5_now, ma20_now = close.rolling(5).mean().iloc[-1], close.rolling(20).mean().iloc[-1]
verdict("均量突破 MaVolume", {
    "5日均线上穿20日(昨下今上)": ma5_prev < ma20_prev and ma5_now > ma20_now,
    f"放量(今量{last['volume']/vol_ma20:.2f} > 1.5×20日均)": last["volume"] > vol_ma20 * 1.5,
})

# 3 高紧旗形
t40, t10 = df.tail(40), df.tail(10)
verdict("高紧旗形 HighTightFlag", {
    f"40日动量({t40['high'].max()/t40['low'].min():.3f} > 1.6)": t40["high"].max() / t40["low"].min() > 1.6,
    f"10日收敛({t10['high'].max()/t10['low'].min():.3f} < 1.15)": t10["high"].max() / t10["low"].min() < 1.15,
    f"高位抗跌(10日低{t10['low'].min():.2f} ≥ 0.8×40日高{t40['high'].max()*0.8:.2f})": t10["low"].min() >= t40["high"].max() * 0.8,
    f"缩量({last['volume']/vol_ma20:.2f} < 0.6)": last["volume"] < vol_ma20 * 0.6,
})

# 4 涨停洗盘
verdict("涨停洗盘 LimitUpShakeout", {
    f"昨日涨停(昨涨{prev['close']/df.iloc[-3]['close']*100-100:+.1f}% ≥ 9.5%)": prev["close"] >= df.iloc[-3]["close"] * 1.095,
    "今日收阴": last["close"] < last["open"],
    f"今日放量(今/昨 {last['volume']/prev['volume']:.2f} > 2.0)": last["volume"] > prev["volume"] * 2.0,
    f"不破昨收(今低{last['low']:.2f} ≥ 昨收{prev['close']:.2f})": last["low"] >= prev["close"],
})

# 5 上升跌停
ma20s, ma60s = close.rolling(20).mean(), close.rolling(60).mean()
verdict("上升趋势跌停 UptrendLimitDown", {
    f"趋势(昨MA20 {ma20s.iloc[-2]:.2f} > 昨MA60 {ma60s.iloc[-2]:.2f})": ma20s.iloc[-2] > ma60s.iloc[-2],
    f"跌停级(今跌{last['close']/prev['close']*100-100:+.1f}% ≤ -9.5%)": last["close"] <= prev["close"] * 0.905,
    f"放量(今量 {last['volume']/vol_ma20:.2f} > 2×20日均)": last["volume"] > vol_ma20 * 2.0,
})

# 6 RPS突破
roll120 = df["high"].rolling(120, min_periods=60).max().iloc[-1]
verdict("RPS突破 RpsBreakout", {
    f"RPS120 {rps120:.1f} ≥ 90": rps120 >= 90,
    f"贴近120日高点(收{last['close']:.2f} ≥ 0.9×{roll120:.2f})": last["close"] >= roll120 * 0.90,
})

# 7 亦霖RPS
h120_prev = df["high"].shift(1).rolling(120, min_periods=120).max().iloc[-1]
verdict("亦霖RPS严格突破 RpsYilin", {
    f"RPS120 {rps120:.1f} ≥ 95": rps120 >= 95,
    f"真突破(收{last['close']:.2f} > 前120日高{h120_prev:.2f})": last["close"] > h120_prev,
    f"放量({last['volume']/vol_ma20:.2f} > 1.5)": last["volume"] > vol_ma20 * 1.5,
    "成交额过亿": last["turnover"] > 1e8,
    "实体阳线且真涨": last["close"] > last["open"] and last["close"] > prev["close"],
    f"趋势(收>MA20 {ma20_now:.2f} > MA60 {ma60s.iloc[-1]:.2f})": last["close"] > ma20_now > ma60s.iloc[-1],
})

# 8 三倍量盘整（信号门 + 盘整门）
t90 = df.tail(90)
vol_ratio = t90["volume"] / t90["volume"].shift(1)
sig = t90[(vol_ratio >= 3) & (t90["close"] > t90["open"]) & (t90["close"] > t90["close"].shift(1))]
has_signal = not sig.empty
sig_days_ago = (len(t90) - 1 - sig.index[-1]) if has_signal else None
verdict("三倍量盘整突破 TripleVolume", {
    f"近90日有三倍量阳线信号(最近{'%d天前' % sig_days_ago if has_signal else '无'})": has_signal,
    f"收盘在MA60上方95%(收{last['close']:.2f} vs 0.95×MA60 {ma60s.iloc[-1]*0.95:.2f})": last["close"] >= ma60s.iloc[-1] * 0.95,
    "成交额过亿": last["turnover"] >= 1e8,
})

# 9 碗口反弹
look15 = df.tail(15)
vr15 = look15["volume"] / look15["volume"].shift(1)
key_candles = look15[(vr15 >= 4) & (look15["close"] > look15["open"])]
max_vol_row = look15.loc[look15["volume"].idxmax()]
key_ok = (not key_candles.empty) and (max_vol_row["close"] > max_vol_row["open"])
verdict("碗口反弹 BowlRebound", {
    f"短期趋势线在上({short_trend:.2f} > {bull_bear:.2f})": short_trend > bull_bear,
    f"KDJ-J低位({j.iloc[-1]:.1f} ≤ 30)": j.iloc[-1] <= 30,
    f"近15日有≥4倍放量阳线且最大量日为阳(信号{len(key_candles)}个)": key_ok,
    f"价格位置(收{last['close']:.2f} 多空线±3%: {bull_bear*0.97:.2f}~{bull_bear*1.03:.2f})": abs(last["close"]/bull_bear - 1) <= 0.03,
    "成交额过亿": last["turnover"] >= 1e8,
})

# 10 定增（事件型）
print("  定增监控 PrivatePlacement: ⚪ 事件型策略（需查近7日定向增发公告，不基于行情指标）")

# 11 30日首次新高
h30 = df["high"].shift(1).rolling(30).max().iloc[-1]
h29 = df["high"].iloc[-30:-1].max()
ma5s, ma10s = close.rolling(5).mean(), close.rolling(10).mean()
m = {"ma5": ma5s, "ma10": ma10s, "ma20": ma20s, "ma60": ma60s}
verdict("30日首次新高 FirstNewHigh30D", {
    f"突破前30日高点(前高{h30:.2f}, 收{last['close']:.2f} ≥ 98%前高)": last["high"] > h30 and last["close"] >= h30 * 0.98,
    f"首次新高(近29日无突破, 29日高{h29:.2f})": h29 < h30,
    f"均线多头(MA5 {ma5s.iloc[-1]:.2f} > MA10 {ma10s.iloc[-1]:.2f} > MA20 {ma20_now:.2f} > MA60 {ma60s.iloc[-1]:.2f})": ma5s.iloc[-1] > ma10s.iloc[-1] > ma20_now > ma60s.iloc[-1],
    f"放量不爆量({last['volume']/vol_ma5:.2f} 于1.2~3.0之间)": 1.2 <= last["volume"]/vol_ma5 <= 3.0,
    "阳线收在上半区且涨幅≤8.5%": last["close"] > last["open"] and (last["close"]-last["low"])/(last["high"]-last["low"]) >= 0.5 and last["close"]/prev["close"] <= 1.085,
    f"成交额>5亿(现{last['turnover']/1e8:.2f}亿)": last["turnover"] > 5e8,
})

# 12 趋势超跌回归
ma20_dev = (ma20_now - last["close"]) / ma20_now
verdict("趋势超跌回归 TrendMeanReversion", {
    f"趋势健康(MA20 {ma20_now:.2f} > MA60 {ma60s.iloc[-1]:.2f}, 收>MA60, 60日涨{pct_60*100:+.1f}%)": ma20_now > ma60s.iloc[-1] and last["close"] > ma60s.iloc[-1] and pct_60 > 0,
    f"低于MA20 3%~10%(现{ma20_dev*100:.1f}%)": 0.03 <= ma20_dev <= 0.10,
    f"近5日回调6%~18%(现{pct_5*100:+.1f}%)": -0.18 < pct_5 <= -0.06,
    "成交额过亿": last["turnover"] > 1e8,
    "止跌确认(阳/高于昨收/长下影)": last["close"] > last["open"] or last["close"] > prev["close"],
})

# 13 止跌候选
draw60 = last["close"] / high60 - 1
pct20 = close.iloc[-1] / close.iloc[-21] - 1
verdict("止跌候选 StopFall(S2/S3)", {
    f"60日回撤≤-12%(现{draw60*100:+.1f}%)": draw60 >= -0.12,
    f"20日收益为负(现{pct20*100:+.1f}%)": pct20 < 0,
    "距前低≤2×ATR(粗算)": (last["close"] - low60) <= 2 * atr14,
})

# 14 三周期
from sequoia_x.strategy.three_cycle import ThreeCycleStrategy
weekly = df.assign(dt=pd.to_datetime(df["date"])).set_index("dt")["close"].resample("W-FRI").last().dropna()
wma20 = weekly.rolling(20).mean()
weekly_ok = bool(len(weekly) >= 24 and weekly.iloc[-1] > wma20.iloc[-1] and wma20.iloc[-1] > wma20.iloc[-4])
flag = ThreeCycleStrategy._flag_pattern(df)
round_b = ThreeCycleStrategy._round_bottom_pattern(df)
conv = ThreeCycleStrategy._convergence_breakout(df)
entry = ThreeCycleStrategy._entry_signal(df)
verdict("三周期(陈凯) ThreeCycle", {
    f"周线多头(周收{weekly.iloc[-1]:.2f} > 20周线{wma20.iloc[-1]:.2f}, 周线上行)": weekly_ok,
    f"特征图形(旗形{'✓' if flag else '✗'}/圆弧{'✓' if round_b else '✗'}/收敛{'✓' if conv else '✗'})": flag or round_b or conv,
    f"切入点(放量{last['volume']/vol_ma20:.2f}>1.5 或 20日新高, 阳线)": entry,
    f"换手率5%~8%(现{last['turn_rate']:.2f}%)": 5.0 <= last["turn_rate"] <= 8.0,
    "成交额过亿": last["turnover"] > 1e8,
})
