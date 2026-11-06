"""ML 策略：预测「未来20个交易日内最高涨幅 ≥15%」的概率。

管线：
  1. 从 SQLite 全量数据构造特征与标签（严格防未来函数：标签只用 t+1..t+20）
  2. 按时间切分 train/valid/test，LightGBM 训练 + 早停
  3. 输出测试集指标（AUC、Top-K 命中率、基准对比）与特征重要性
  4. 保存模型，并对最新交易日全市场扫描输出高概率股票

用法：.venv/bin/python scripts/train_surge_model.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import get_settings

HORIZON = 20          # 预测窗口（交易日）
TARGET_RET = 0.15     # 目标涨幅
MIN_BARS = 130        # 每只股票最少K线数

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


def build_dataset(db_path: str) -> pd.DataFrame:
    t0 = time.time()
    print("[1/4] 读取全量数据...", flush=True)
    import sqlite3

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT symbol, date, open, high, low, close, volume, turnover, turn_rate"
        " FROM stock_daily",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume", "turnover", "turn_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby("symbol", group_keys=False)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    print("[2/4] 计算时序特征...", flush=True)
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

    print("[3/4] 计算截面 RPS + 未来标签...", flush=True)
    for p in (20, 60, 120):
        r = close / g["close"].shift(p) - 1
        df[f"rps_{p}"] = r.groupby(df["date"]).rank(pct=True) * 100

    # 未来20日最高涨幅（只含 t+1..t+20）：
    # 反转序列 rolling(20).max() 得到 max(close[t..t+19])，再 shift(-1) 即 max(close[t+1..t+20])
    fut_max = (
        close[::-1]
        .groupby(df["symbol"][::-1], group_keys=False)
        .rolling(HORIZON, min_periods=HORIZON)
        .max()[::-1]
        .reset_index(drop=True)
        .shift(-1)
    )
    df["fut_ret"] = fut_max / close - 1
    df["label"] = (df["fut_ret"] >= TARGET_RET).astype(int)

    df = df.dropna(subset=FEATURES + ["label"])
    print(f"[4/4] 数据集: {len(df)} 样本, 正样本率 {df['label'].mean()*100:.2f}%, 耗时 {time.time()-t0:.0f}s", flush=True)
    return df


def main() -> None:
    settings = get_settings()
    df = build_dataset(settings.db_path)

    train = df[df["date"] < "2025-06-01"]
    valid = df[(df["date"] >= "2025-06-01") & (df["date"] < "2026-01-01")]
    test = df[df["date"] >= "2026-01-01"]
    print(f"切分: train {len(train)} / valid {len(valid)} / test {len(test)}", flush=True)
    if valid.empty or test.empty:
        print("数据不足，无法切分")
        sys.exit(1)

    import numpy as np
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    # 训练集抽样控制规模（HGB 直方图算法，80万样本即可稳定）
    if len(train) > 800_000:
        train = train.sample(800_000, random_state=42)
    print(f"训练样本(抽样后): {len(train)}", flush=True)

    print("开始训练 HistGradientBoosting...", flush=True)
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        scoring="roc_auc",
        random_state=42,
    )
    model.fit(train[FEATURES], train["label"])
    print(f"训练完成，耗时 {time.time()-t0:.0f}s，实际迭代 {model.n_iter_} 轮", flush=True)

    # ── 评估 ──
    yt = test["label"].values
    pt = model.predict_proba(test[FEATURES])[:, 1]
    auc = roc_auc_score(yt, pt)
    base_rate = yt.mean()
    print(f"\n=== 测试集评估（{test['date'].min().date()} ~ {test['date'].max().date()}，{len(test)} 样本）===")
    print(f"基准（随机选股）正样本率: {base_rate*100:.2f}%")
    print(f"AUC: {auc:.4f}")

    top_fracs = [0.01, 0.03, 0.05, 0.10]
    print("Top-K 命中率（按预测概率排序）:")
    order = np.argsort(-pt)
    for f in top_fracs:
        k = max(1, int(len(test) * f))
        hit = yt[order[:k]].mean()
        print(f"  Top {f*100:>4.0f}% ({k:>6} 只): 命中率 {hit*100:5.2f}%  (相对基准 {hit/base_rate:.2f} 倍)")

    for thr in (0.3, 0.4, 0.5):
        mask = pt >= thr
        if mask.sum() == 0:
            continue
        print(f"  概率≥{thr}: 命中率 {yt[mask].mean()*100:5.2f}% ({mask.sum()} 只)")

    # 特征重要性（置换重要性，抽样 8000 条加速）
    imp_sample = test.sample(min(8000, len(test)), random_state=42)
    perm = permutation_importance(
        model, imp_sample[FEATURES], imp_sample["label"],
        n_repeats=3, random_state=42, n_jobs=-1,
    )
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    print("\n特征重要性 Top10:")
    print(imp.head(10).round(4).to_string())

    # ── 保存模型 ──
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "surge20_model.joblib"
    dump(model, model_path)
    (out_dir / "surge20_features.txt").write_text("\n".join(FEATURES), encoding="utf-8")
    (out_dir / "surge20_eval.txt").write_text(
        f"auc={auc:.4f}\nbase_rate={base_rate:.4f}\nsamples={len(test)}\n"
        f"train_until={train['date'].max().date()}\ntest_from={test['date'].min().date()}\n"
        + imp.head(10).round(4).to_string(),
        encoding="utf-8",
    )
    print(f"\n模型已保存: {model_path}")

    # ── 最新交易日全市场扫描 ──
    latest_date = df["date"].max()
    scan = df[df["date"] == latest_date].copy()
    scan["prob"] = model.predict_proba(scan[FEATURES])[:, 1]
    scan = scan.sort_values("prob", ascending=False)
    try:
        import akshare as ak
        names = ak.stock_info_a_code_name()
        name_map = dict(zip(names["code"].astype(str).str.zfill(6), names["name"]))
        scan["name"] = scan["symbol"].map(name_map)
    except Exception as exc:
        print(f"名称获取失败（不影响主流程）: {exc}")
        scan["name"] = ""
    top = scan.head(50)
    top.to_csv(out_dir / f"surge20_scan_{latest_date.date()}.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 最新交易日 {latest_date.date()} 扫描（Top 20 高概率）===")
    for _, r in top.head(20).iterrows():
        print(
            f"  {r['symbol']} {r['name']}  概率 {r['prob']*100:5.1f}%  "
            f"RPS120 {r['rps_120']:.0f}  20日动量 {r['ret_20']*100:+.1f}%  换手 {r['turn_rate']:.1f}%"
        )


if __name__ == "__main__":
    main()
