"""采集每日全市场 isST 标志到 st_flag 表（断点续传）。

用法：.venv/bin/python scripts/fetch_st_flags.py
"""
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import baostock as bs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import get_settings

settings = get_settings()
conn = sqlite3.connect(settings.db_path)
conn.execute(
    "CREATE TABLE IF NOT EXISTS st_flag ("
    "symbol TEXT NOT NULL, date TEXT NOT NULL, isST INTEGER,"
    " PRIMARY KEY (symbol, date))"
)
conn.execute("CREATE TABLE IF NOT EXISTS st_meta (day TEXT PRIMARY KEY)")
conn.commit()

lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    sys.exit(1)

today = date.today().strftime("%Y-%m-%d")
rs_cal = bs.query_trade_dates(start_date=settings.start_date, end_date=today)
days = []
while rs_cal.next():
    row = rs_cal.get_row_data()
    if row[1] == "1":
        days.append(row[0])

done = {r[0] for r in conn.execute("SELECT day FROM st_meta")}
todo = [d for d in days if d not in done]
print(f"共 {len(days)} 个交易日，已完成 {len(done)} 天，待采集 {len(todo)} 天", flush=True)

ok = fail = 0
for i, d in enumerate(todo):
    got = False
    for attempt in range(3):
        try:
            k_rs = bs.query_daily_history_k_AStock(date=d)
            if k_rs.error_code != "0":
                raise RuntimeError(k_rs.error_msg)
            fields = list(k_rs.fields)
            ist = fields.index("isST")
            rows = []
            while k_rs.next():
                r = k_rs.get_row_data()
                code = r[fields.index("code")].lower()
                symbol = code.split(".")[-1]
                rows.append((symbol, d, int(r[ist] or 0)))
            conn.executemany(
                "INSERT OR REPLACE INTO st_flag (symbol, date, isST) VALUES (?,?,?)",
                rows,
            )
            conn.execute("INSERT OR REPLACE INTO st_meta (day) VALUES (?)", (d,))
            conn.commit()
            ok += 1
            got = True
            break
        except Exception as exc:
            if attempt < 2:
                print(f"[{d}] 第{attempt + 1}次失败: {exc}", flush=True)
                time.sleep(2 ** (attempt + 1))
                bs.logout()
                time.sleep(1)
                bs.login()
            else:
                print(f"[{d}] 3次失败，跳过", flush=True)
    if not got:
        fail += 1
    if (i + 1) % 100 == 0:
        print(f"进度 {i + 1}/{len(todo)}，成功 {ok} 失败 {fail}", flush=True)
    time.sleep(0.2)

bs.logout()
conn.close()
print(f"完成：成功 {ok} 天，失败 {fail} 天", flush=True)
