"""执行方案矩阵回测：入场时点 × 止损 × 止盈 × Top分数。

用法：.venv/bin/python scripts/backtest_exec.py
前置：非ST模型已训练（data/surge20_model_nost.joblib）+ st_flag 表已采集
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import get_settings

from train_surge_model import FEATURES, HORIZON, TARGET_RET, build_dataset

MIN_FWD_BARS = 15


def _gen_signals(test: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    parts = []
    for d, grp in test.groupby("date"):
        k = max(1, int(len(grp) * top_frac))
        parts.append(grp.nlargest(k, "prob")[["symbol", "date"]])
    return pd.concat(parts)


def _simulate(df_sig, ohlc, entry_mode, stop_pct=None, tp_pct=None):
    """返回统计字典。

    entry_mode: 'next_open' = T+1 开盘入场；'signal_close' = 信号日收盘入场
    """
    rets, helds = [], []
    mfe_hit = 0
    n = 0
    for _, row in df_sig.iterrows():
        sym, d = row["symbol"], row["date"]
        g = ohlc[sym]
        pos = g.index[g["date"] == d]
        if len(pos) == 0:
            continue
        i = pos[0]
        fwd = g.iloc[i + 1: i + 1 + HORIZON]
        if len(fwd) < MIN_FWD_BARS:
            continue
        if entry_mode == "signal_close":
            entry = float(g.iloc[i]["close"])
        else:
            entry = float(fwd.iloc[0]["open"])
        if entry <= 0:
            continue
        stop = entry * (1 - stop_pct) if stop_pct else -np.inf
        tp = entry * (1 + tp_pct) if tp_pct else np.inf
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
        # 曾触及 +15%（最高价口径）
        if fwd["high"].max() / entry - 1 >= TARGET_RET - 1e-9:
            mfe_hit += 1
        rets.append(exit_px / entry - 1)
        helds.append(held)
        n += 1
    rets = np.array(rets)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    return {
        "n": n,
        "hit15": (rets >= TARGET_RET - 1e-9).mean() if n else 0,
        "mfe_hit15": mfe_hit / n if n else 0,
        "win": (rets > 0).mean() if n else 0,
        "avg": rets.mean() if n else 0,
        "med": np.median(rets) if n else 0,
        "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": losses.mean() if len(losses) else 0,
        "pf": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf"),
        "hold": float(np.mean(helds)) if helds else 0,
    }


def main() -> None:
    settings = get_settings()
    df = build_dataset(settings.db_path)
    conn = sqlite3.connect(settings.db_path)
    st_df = pd.read_sql("SELECT symbol, date FROM st_flag WHERE isST=1", conn)
    conn.close()
    st_df["isST"] = 1
    df["dstr"] = df["date"].dt.strftime("%Y-%m-%d")
    st_df["dstr"] = st_df["date"]
    df = df.merge(st_df[["symbol", "dstr", "isST"]], on=["symbol", "dstr"], how="left")
    df["isST"] = df["isST"].fillna(0).astype(int)
    df = df[df["isST"] == 0].reset_index(drop=True)

    from joblib import load
    model = load("data/surge20_model_nost.joblib")

    test = df[df["date"] >= "2026-01-01"].copy()
    test["prob"] = model.predict_proba(test[FEATURES])[:, 1]

    ohlc = {}
    for sym, grp in df.groupby("symbol"):
        g = grp.sort_values("date")
        ohlc[sym] = g[["date", "open", "high", "low", "close"]].reset_index(drop=True)

    variants = [
        ("T+1开盘", "无止损", "无止盈", "Top5%", "next_open", None, None, 0.05),
        ("T+1开盘", "-10%", "+15%", "Top5%", "next_open", 0.10, 0.15, 0.05),
        ("T+1开盘", "-15%", "无", "Top5%", "next_open", 0.15, None, 0.05),
        ("T+1开盘", "-20%", "无", "Top5%", "next_open", 0.20, None, 0.05),
        ("收盘入场", "无", "无", "Top5%", "signal_close", None, None, 0.05),
        ("收盘入场", "-15%", "无", "Top5%", "signal_close", 0.15, None, 0.05),
        ("收盘入场", "-10%", "+15%", "Top5%", "signal_close", 0.10, 0.15, 0.05),
        ("T+1开盘", "-15%", "无", "Top1%", "next_open", 0.15, None, 0.01),
        ("收盘入场", "-15%", "无", "Top1%", "signal_close", 0.15, None, 0.01),
        # 概率阈值型信号（捕捉高置信期）
        ("收盘入场", "无", "无", "P≥0.7", "signal_close", None, None, "thr07"),
        ("收盘入场", "无", "无", "P≥0.6", "signal_close", None, None, "thr06"),
        ("T+1开盘", "无", "无", "P≥0.7", "next_open", None, None, "thr07"),
        # 对照：全局 Top5%（不可实盘，仅供对照）
        ("收盘入场", "无", "无", "全局5%", "signal_close", None, None, "global05"),
    ]

    sig_cache = {}
    print(f"{'入场':<8}{'止损':<7}{'止盈':<7}{'信号':<7}{'交易':>7}{'达标15%':>8}{'曾触15%':>8}{'胜率':>7}{'平均':>8}{'中位':>8}{'盈亏比':>7}{'持仓天':>7}")
    for name, sl_name, tp_name, sig_name, emode, sl, tp, frac in variants:
        key = (emode, frac)
        if key not in sig_cache:
            if frac == "global05":
                sig = test.nlargest(int(len(test) * 0.05), "prob")[["symbol", "date", "prob"]]
            elif frac in ("thr07", "thr06"):
                thr = 0.7 if frac == "thr07" else 0.6
                sig = test[test["prob"] >= thr][["symbol", "date", "prob"]]
            else:
                sig = _gen_signals(test, frac).merge(
                    test[["symbol", "date", "prob"]], on=["symbol", "date"], how="inner"
                )
            sig_cache[key] = sig
        s = _simulate(sig_cache[key], ohlc, emode, sl, tp)
        print(
            f"{name:<8}{sl_name:<7}{tp_name:<7}{sig_name:<7}{s['n']:>7}"
            f"{s['hit15']*100:>7.1f}%{s['mfe_hit15']*100:>7.1f}%{s['win']*100:>6.1f}%"
            f"{s['avg']*100:>7.2f}%{s['med']*100:>7.2f}%{s['pf']:>7.2f}{s['hold']:>6.1f}"
        )


if __name__ == "__main__":
    main()
