"""非 ST 版 20日大涨模型 + 止损回测。

用法：.venv/bin/python scripts/train_nost_backtest.py
前置：st_flag 表已采集（scripts/fetch_st_flags.py）
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import get_settings

from train_surge_model import FEATURES, HORIZON, TARGET_RET, build_dataset

STOP_LOSSES = [0.08, 0.10, 0.12]
TOP_FRAC = 0.05
MIN_FWD_BARS = 15  # 信号日之后至少需要这么多根K线才算有效交易


def main() -> None:
    settings = get_settings()
    df = build_dataset(settings.db_path)

    # ── ST 过滤 ──
    conn = sqlite3.connect(settings.db_path)
    st_df = pd.read_sql("SELECT symbol, date FROM st_flag WHERE isST=1", conn)
    conn.close()
    st_df["isST"] = 1
    df["dstr"] = df["date"].dt.strftime("%Y-%m-%d")
    st_df["dstr"] = st_df["date"]
    df = df.merge(st_df[["symbol", "dstr", "isST"]], on=["symbol", "dstr"], how="left")
    df["isST"] = df["isST"].fillna(0).astype(int)
    n_st = int(df["isST"].sum())
    print(f"样本 {len(df)}，其中 ST 样本 {n_st}（{n_st/len(df)*100:.1f}%）已剔除", flush=True)
    df = df[df["isST"] == 0].reset_index(drop=True)

    train = df[df["date"] < "2025-06-01"]
    valid = df[(df["date"] >= "2025-06-01") & (df["date"] < "2026-01-01")]
    test = df[df["date"] >= "2026-01-01"]
    print(f"切分: train {len(train)} / valid {len(valid)} / test {len(test)}", flush=True)

    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    if len(train) > 800_000:
        train = train.sample(800_000, random_state=42)

    print("训练非 ST 模型...", flush=True)
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=1000, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=50,
        scoring="roc_auc", random_state=42,
    )
    model.fit(train[FEATURES], train["label"])
    print(f"训练完成 {time.time()-t0:.0f}s，迭代 {model.n_iter_} 轮", flush=True)

    yt = test["label"].values
    pt = model.predict_proba(test[FEATURES])[:, 1]
    auc = roc_auc_score(yt, pt)
    base_rate = yt.mean()
    print(f"\n=== 非ST模型测试集评估（{len(test)} 样本）===")
    print(f"基准命中率: {base_rate*100:.2f}%  AUC: {auc:.4f}")
    order = np.argsort(-pt)
    for f in (0.01, 0.05, 0.10):
        k = max(1, int(len(test) * f))
        hit = yt[order[:k]].mean()
        print(f"  Top {f*100:>4.0f}%: 命中率 {hit*100:5.2f}% ({hit/base_rate:.2f} 倍)")

    dump(model, "data/surge20_model_nost.joblib")
    print("模型已保存: data/surge20_model_nost.joblib", flush=True)

    # ── 止损回测（测试期冻结模型，T+1 开盘入场）──
    print("\n=== 止损回测（测试期 2026-01 ~，信号=当日概率 Top 5%，T+1 开盘入场）===", flush=True)
    test = test.copy()
    test["prob"] = pt

    # 每日截面 Top 5%
    signals = []
    for d, grp in test.groupby("date"):
        k = max(1, int(len(grp) * TOP_FRAC))
        sig = grp.nlargest(k, "prob")[["symbol", "date"]]
        signals.append(sig)
    signals = pd.concat(signals)

    # 为信号匹配未来窗口（每只股票的 t+1..t+20 OHLC）
    df_sig = df.merge(
        signals.assign(has_signal=1),
        on=["symbol", "date"], how="inner",
    )
    print(f"信号数: {len(df_sig)}", flush=True)

    # 每信号构建前向序列（向量化：按 symbol 分组，shift 出 t+1..t+20 的开高低收）
    # 简化：对每个信号用 numpy 前向模拟
    ohlc = {}
    for sym, grp in df.groupby("symbol"):
        g = grp.sort_values("date")
        ohlc[sym] = g[["date", "open", "high", "low", "close"]].reset_index(drop=True)

    results = {}
    for sl in STOP_LOSSES:
        stats = _simulate(df_sig, ohlc, sl)
        results[sl] = stats

    print(f"\n{'止损':<8}{'交易数':>7}{'达标15%':>9}{'盈利':>7}{'平均收益':>9}{'中位':>8}{'平均盈利':>9}{'平均亏损':>9}{'盈亏比':>7}")
    for sl, s in results.items():
        print(
            f"{-sl*100:>5.0f}%  {s['n']:>7}{s['hit_rate']*100:>8.1f}%{s['win_rate']*100:>7.1f}%"
            f"{s['avg_ret']*100:>8.2f}%{s['median']*100:>7.2f}%"
            f"{s['avg_win']*100:>8.2f}%{s['avg_loss']*100:>8.2f}%{s['pf']:>7.2f}"
        )

    # 组合曲线（等权，全部信号）
    sl = 0.10
    curve = _equity_curve(df_sig, ohlc, sl)
    final = curve[-1]
    peak = np.maximum.accumulate(curve)
    mdd = float(((curve - peak) / peak).min())
    print(f"\n等权组合曲线（止损-10%，T+1开盘入场）:")
    print(f"  期末净值: {final:.3f}（初始 1.0）  最大回撤: {mdd*100:.1f}%  曲线长度: {len(curve)} 天")

    # ── 最新日扫描（非ST模型）──
    latest_date = df["date"].max()
    scan = df[df["date"] == latest_date].copy()
    scan["prob"] = model.predict_proba(scan[FEATURES])[:, 1]
    scan = scan.sort_values("prob", ascending=False).head(30)
    scan.to_csv(f"data/surge20_scan_nost_{latest_date.date()}.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 最新交易日 {latest_date.date()} 非ST模型扫描 Top 15 ===")
    for _, r in scan.head(15).iterrows():
        print(f"  {r['symbol']}  概率 {r['prob']*100:5.1f}%  20日动量 {r['ret_20']*100:+.1f}%  换手 {r['turn_rate']:.1f}%")


def _simulate(df_sig: pd.DataFrame, ohlc: dict, sl: float) -> dict:
    """逐信号模拟：T+1 开盘入场；止损价=entry*(1-sl)；止盈价=entry*1.15；20日收盘出场。"""
    rets = []
    hit = 0
    days_held = []
    for _, row in df_sig.iterrows():
        sym = row["symbol"]
        d = row["date"]
        g = ohlc[sym]
        pos = g.index[g["date"] == d]
        if len(pos) == 0:
            continue
        i = pos[0]
        fwd = g.iloc[i + 1: i + 1 + HORIZON]
        if len(fwd) < MIN_FWD_BARS:
            continue
        entry = float(fwd.iloc[0]["open"])
        if entry <= 0:
            continue
        stop = entry * (1 - sl)
        tp = entry * (1 + TARGET_RET)
        exit_px = None
        held = 0
        for j in range(len(fwd)):
            bar = fwd.iloc[j]
            held = j + 1
            if bar["low"] <= stop:
                exit_px = min(stop, bar["open"])
                break
            if bar["high"] >= tp:
                exit_px = tp
                break
        if exit_px is None:
            exit_px = float(fwd.iloc[-1]["close"])
            held = len(fwd)
        ret = exit_px / entry - 1
        rets.append(ret)
        days_held.append(held)
        if ret >= TARGET_RET - 1e-9:
            hit += 1
    rets = np.array(rets)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    return {
        "n": len(rets),
        "hit_rate": hit / len(rets) if len(rets) else 0,
        "win_rate": (rets > 0).mean() if len(rets) else 0,
        "avg_ret": rets.mean() if len(rets) else 0,
        "median": np.median(rets) if len(rets) else 0,
        "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": losses.mean() if len(losses) else 0,
        "pf": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf"),
        "avg_hold": np.mean(days_held) if days_held else 0,
    }


def _equity_curve(df_sig: pd.DataFrame, ohlc: dict, sl: float) -> np.ndarray:
    """等权组合净值曲线：每天取所有存续交易的平均价值（每笔交易初始 1.0）。"""
    trades = []
    for _, row in df_sig.iterrows():
        sym = row["symbol"]
        d = row["date"]
        g = ohlc[sym]
        pos = g.index[g["date"] == d]
        if len(pos) == 0:
            continue
        i = pos[0]
        fwd = g.iloc[i + 1: i + 1 + HORIZON]
        if len(fwd) < MIN_FWD_BARS:
            continue
        entry = float(fwd.iloc[0]["open"])
        if entry <= 0:
            continue
        trades.append((entry, fwd))
    if not trades:
        return np.array([1.0])

    all_dates = sorted({d for _, f in trades for d in f["date"]})
    didx = {d: i for i, d in enumerate(all_dates)}
    n = len(all_dates)
    total = np.zeros(n)
    cnt = np.zeros(n)
    for entry, fwd in trades:
        stop = entry * (1 - sl)
        tp = entry * (1 + TARGET_RET)
        for _, bar in fwd.iterrows():
            di = didx[bar["date"]]
            if bar["low"] <= stop:
                v = min(stop, bar["open"]) / entry
            elif bar["high"] >= tp:
                v = tp / entry
            else:
                v = bar["close"] / entry
            total[di] += v
            cnt[di] += 1
            if bar["low"] <= stop or bar["high"] >= tp:
                break
    eq = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    eq = pd.Series(eq).ffill().fillna(1.0).values
    return eq


if __name__ == "__main__":
    main()
