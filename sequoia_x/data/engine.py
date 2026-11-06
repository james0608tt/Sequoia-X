"""数据引擎模块：负责 SQLite 行情数据存储与 baostock/akshare 数据同步。"""

import socket as _socket
import sqlite3
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def _sleep_with_jitter(base_seconds: float, jitter_seconds: float = 0.0) -> None:
    """在 baostock 请求之间暂停，避免突发流量触发限流。"""
    if base_seconds <= 0 and jitter_seconds <= 0:
        return

    import random
    import time

    wait_seconds = max(0.0, base_seconds)
    if jitter_seconds > 0:
        wait_seconds += random.uniform(0.0, jitter_seconds)

    if wait_seconds > 0:
        time.sleep(wait_seconds)


# ── 强制 IPv4：部分网络下 IPv6 出口到东财 CDN 会卡死 TLS 握手
#    （TCP 建立成功但服务器永不完成握手，urllib3 对握手阶段无超时，
#    导致进程永久挂起）。此处将 getaddrinfo 过滤为仅 IPv4。
_orig_getaddrinfo = _socket.getaddrinfo


def _getaddrinfo_ipv4_only(host: str, *args, **kwargs):
    results = _orig_getaddrinfo(host, *args, **kwargs)
    v4 = [r for r in results if r[0] == _socket.AF_INET]
    return v4 or results


_socket.getaddrinfo = _getaddrinfo_ipv4_only


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_daily (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT    NOT NULL,
    date      TEXT    NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    turnover  REAL,
    turn_rate REAL,
    UNIQUE (symbol, date)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_daily (symbol, date);
"""

_CREATE_FACTOR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS adjust_factor (
    code             TEXT NOT NULL,
    dividOperateDate TEXT NOT NULL,
    backAdjustFactor REAL,
    PRIMARY KEY (code, dividOperateDate)
);
"""

_CREATE_FACTOR_META_SQL = """
CREATE TABLE IF NOT EXISTS factor_meta (
    day TEXT PRIMARY KEY
);
"""

# 旧库升级：为已存在的 stock_daily 表补充 turn_rate 列
_ALTER_TABLE_SQL = "ALTER TABLE stock_daily ADD COLUMN turn_rate REAL"


def _bs_fetch_batch(tasks: list) -> list:
    """多进程 worker：独立 login，批量拉取 baostock 数据。"""
    import baostock as bs
    bs.login()
    results = []
    for symbol, bs_code, start, end in tasks:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="1",  # 后复权
        )
        if rs.error_code != "0":
            continue
        while rs.next():
            results.append([symbol] + rs.get_row_data())
    bs.logout()
    return results


class DataEngine:
    """行情数据引擎，负责 SQLite 存储和 baostock 数据同步。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        # baostock 请求限速参数（V3 升级）
        self.baostock_request_delay_seconds: float = settings.baostock_request_delay_seconds
        self.baostock_request_jitter_seconds: float = settings.baostock_request_jitter_seconds
        # 全市场 OHLCV 内存缓存（惰性填充）
        self._ohlcv_cache: dict[str, pd.DataFrame] | None = None
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # WAL 模式：读写并发更友好
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.execute(_CREATE_FACTOR_TABLE_SQL)
            conn.execute(_CREATE_FACTOR_META_SQL)
            # 兼容旧库：若无 turn_rate 列则补上
            cols = {r[1] for r in conn.execute("PRAGMA table_info(stock_daily)").fetchall()}
            if "turn_rate" not in cols:
                conn.execute(_ALTER_TABLE_SQL)
            conn.commit()
        logger.info(f"数据库初始化完成：{self.db_path}")

    def _get_last_date(self, symbol: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def load_all_ohlcv(self) -> dict[str, pd.DataFrame]:
        """一次性返回全市场日线数据（按 symbol 分组）。

        复用内存缓存；首次调用时从 SQLite 填充缓存。
        返回的 dict 引用缓存内的 DataFrame，调用方可原地修改（列追加等）。
        """
        self._ensure_cache()
        assert self._ohlcv_cache is not None
        return self._ohlcv_cache

    def _load_from_db(self) -> dict[str, pd.DataFrame]:
        """从 SQLite 一次性读取全表并按 symbol 分组。"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT symbol,date,open,high,low,close,volume,turnover,turn_rate "
                "FROM stock_daily ORDER BY symbol, date",
                conn,
            )
        if df.empty:
            return {}
        result: dict[str, pd.DataFrame] = {}
        for sym, grp in df.groupby("symbol", sort=False):
            result[str(sym)] = grp.reset_index(drop=True)
        return result

    def _ensure_cache(self) -> None:
        """惰性填充内存缓存（如已存在则跳过）。"""
        if self._ohlcv_cache is not None:
            return
        self._ohlcv_cache = self._load_from_db()
        logger.info(
            "OHLCV 缓存已就绪: %d 只股票",
            len(self._ohlcv_cache),
        )

    def invalidate_cache(self) -> None:
        """强制清空内存缓存，下次 get_ohlcv / load_all_ohlcv 将重新读取数据库。"""
        self._ohlcv_cache = None

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        """读取单只股票日线数据（优先命中内存缓存，未命中则直接查库）。"""
        if self._ohlcv_cache is not None and symbol in self._ohlcv_cache:
            return self._ohlcv_cache[symbol].copy()
        self._ensure_cache()
        if self._ohlcv_cache is not None and symbol in self._ohlcv_cache:
            return self._ohlcv_cache[symbol].copy()
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT * FROM stock_daily WHERE symbol = ? ORDER BY date",
                conn,
                params=(symbol,),
            )
        return df

    def pause_between_baostock_requests(self) -> None:
        """按配置的限速参数在 baostock 请求间暂停（降低限流风险）。"""
        _sleep_with_jitter(
            self.baostock_request_delay_seconds,
            self.baostock_request_jitter_seconds,
        )

    @staticmethod
    def _to_baostock_code(symbol: str) -> str:
        """将纯数字代码转为 baostock 格式：6/9开头 -> sh，其余 -> sz。"""
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        return f"{prefix}.{symbol}"

    # ── 数据同步 ──

    # ── 官方批量接口（新版 baostock 客户端，需含每日批量接口）──
    # 官方文档：https://www.baostock.com/mainContent?file=DailyUpdates.md
    # 思路：逐交易日 2 次请求（全市场日K + 每日复权因子）→ 本地按官方公式
    #      （不复权价 × backAdjustFactor）计算后复权 → 入库。
    #      请求量从逐股 5223 次降为 ~N天×2 次，从根本上规避限流。

    @staticmethod
    def _bs_bulk_available() -> bool:
        """检测新版 baostock 客户端（含每日批量接口）是否可用。"""
        import baostock as bs
        return hasattr(bs, "query_daily_history_k_AStock") and hasattr(
            bs, "query_daily_adjust_factor")

    def _bulk_fetch_day(self, bs, day: str) -> tuple:
        """批量拉取某日全市场日K与每日复权因子。

        Returns:
            (k_rows, k_fields, f_rows, f_fields)：原始行列表与字段名列表。
        """
        k_rs = bs.query_daily_history_k_AStock(date=day)
        if k_rs.error_code != "0":
            raise RuntimeError(f"批量日K失败: {k_rs.error_msg}")
        k_rows = []
        while k_rs.next():
            k_rows.append(k_rs.get_row_data())
        k_fields = list(k_rs.fields)

        f_rs = bs.query_daily_adjust_factor(date=day)
        if f_rs.error_code != "0":
            raise RuntimeError(f"每日复权因子失败: {f_rs.error_msg}")
        f_rows = []
        while f_rs.next():
            f_rows.append(f_rs.get_row_data())
        f_fields = list(f_rs.fields)
        return k_rows, k_fields, f_rows, f_fields

    def _factor_day_done(self, day: str) -> bool:
        """该交易日的复权因子是否已拉取过（断点续传标记）。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM factor_meta WHERE day = ?", (day,)
            ).fetchone()
        return row is not None

    def _mark_factor_day(self, day: str) -> None:
        """标记该交易日的复权因子已拉取完成。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO factor_meta (day) VALUES (?)", (day,)
            )
            conn.commit()

    def _bulk_store_factors(self, f_rows: list, f_fields: list) -> None:
        """复权因子事件持久化到 adjust_factor 表（供每日同步重建因子缓存）。"""
        if not f_rows:
            return
        f_code_i = f_fields.index("code") if "code" in f_fields else 0
        f_date_i = (f_fields.index("dividOperateDate")
                    if "dividOperateDate" in f_fields else 1)
        f_back_i = (f_fields.index("backAdjustFactor")
                    if "backAdjustFactor" in f_fields else None)
        if f_back_i is None:
            return
        rows = []
        for r in f_rows:
            try:
                rows.append((str(r[f_code_i]).lower(), r[f_date_i], float(r[f_back_i])))
            except (ValueError, IndexError):
                continue
        if rows:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO adjust_factor"
                    "(code, dividOperateDate, backAdjustFactor) VALUES (?,?,?)",
                    rows,
                )
                conn.commit()

    def _has_factor_table(self) -> bool:
        """adjust_factor 表是否有数据（批量回填后才为 True，供增量同步判断）。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM adjust_factor").fetchone()
        return bool(row and row[0])

    def _bulk_day_to_records(self, k_rows: list, k_fields: list, day: str) -> list:
        """将某日批量原始数据按官方公式本地算后复权，产出入库记录。

        官方公式（已验证与服务器后复权一致）：
            后复权价 = 不复权价 × 该股当日适用的 backAdjustFactor

        复权因子直接查 adjust_factor 表（每代码取 dividOperateDate ≤ 当日
        的最新因子），与运行顺序无关，天然支持断点续传。
        """
        idx = {name: k_fields.index(name) for name in
               ("date", "code", "open", "high", "low", "close", "volume", "amount", "turn")}

        factor_cache: dict[str, float] = {}
        conn = sqlite3.connect(self.db_path)
        try:
            def factor_for(code: str) -> float:
                if code in factor_cache:
                    return factor_cache[code]
                row = conn.execute(
                    "SELECT backAdjustFactor FROM adjust_factor"
                    " WHERE code = ? AND dividOperateDate <= ?"
                    " ORDER BY dividOperateDate DESC LIMIT 1",
                    (code, day),
                ).fetchone()
                fac = float(row[0]) if row and row[0] is not None else 1.0
                factor_cache[code] = fac
                return fac

            recs = []
            for r in k_rows:
                try:
                    code = str(r[idx["code"]]).lower()
                    symbol = code.split(".")[-1]
                    fac = factor_for(code)
                    # 换手率：停牌日为空字符串，按官方文档置 0
                    turn_str = str(r[idx["turn"]]).strip()
                    turn_rate = float(turn_str) if turn_str else 0.0
                    recs.append((
                        symbol,
                        r[idx["date"]],
                        float(r[idx["open"]]) * fac,
                        float(r[idx["high"]]) * fac,
                        float(r[idx["low"]]) * fac,
                        float(r[idx["close"]]) * fac,
                        float(r[idx["volume"]]),
                        float(r[idx["amount"]]),
                        turn_rate,
                    ))
                except (ValueError, IndexError, KeyError):
                    continue
            return recs
        finally:
            conn.close()

    def _bulk_write_day(self, day: str, recs: list) -> int:
        """按日覆盖写入 SQLite（先删当日再插入）。"""
        if not recs:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM stock_daily WHERE date = ?", (day,))
            conn.executemany(
                "INSERT OR REPLACE INTO stock_daily"
                "(symbol, date, open, high, low, close, volume, turnover, turn_rate)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                recs,
            )
            conn.commit()
        return len(recs)

    def backfill_bulk(self) -> None:
        """官方批量接口回填：逐交易日 2 次请求覆盖全市场，本地算后复权后写入。

        因子处理（官方要求）：
        - 起始日期前的复权因子事件先预载（官方文档：需取更早因子，从 2020 年起）
        - 因子拉取按天标记（factor_meta），K线按天入库，双断点续传
        - 已入库的交易日只补缺失的因子，缺失日才拉日K并写入
        """
        import time
        from datetime import date

        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return

        today = date.today().strftime("%Y-%m-%d")

        def _trade_days(start: str, end: str) -> list[str]:
            rs = bs.query_trade_dates(start_date=start, end_date=end)
            days = []
            while rs.next():
                row = rs.get_row_data()
                if row[1] == "1":  # is_trading_day
                    days.append(row[0])
            return days

        def _fetch_factors(day: str) -> tuple:
            f_rs = bs.query_daily_adjust_factor(date=day)
            if f_rs.error_code != "0":
                raise RuntimeError(f"每日复权因子失败: {f_rs.error_msg}")
            f_rows = []
            while f_rs.next():
                f_rows.append(f_rs.get_row_data())
            return f_rows, list(f_rs.fields)

        def _with_retry(fn, day: str, what: str):
            for attempt in range(3):
                try:
                    return fn(day)
                except Exception as exc:
                    if attempt < 2:
                        logger.warning(f"[{day}] {what}第{attempt + 1}次失败: {exc}，重试中")
                        time.sleep(2 ** (attempt + 1))
                        bs.logout()
                        time.sleep(1)
                        bs.login()
                    else:
                        logger.warning(f"[{day}] {what} 3次重试均失败，跳过")
            return None

        # ── 1. 种子因子：官方要求取更早的复权因子，从 2020 年起预载 ──
        seed_start = "2020-01-01"
        seed_days = _trade_days(seed_start, self.start_date)
        pending_seed = [d for d in seed_days if not self._factor_day_done(d)]
        logger.info(
            f"预载历史复权因子（{seed_start} ~ {self.start_date}）："
            f"共 {len(seed_days)} 天，待拉取 {len(pending_seed)} 天"
        )
        for i, d in enumerate(pending_seed):
            res = _with_retry(_fetch_factors, d, "因子预载")
            if res is not None:
                self._bulk_store_factors(res[0], res[1])
                self._mark_factor_day(d)
            if (i + 1) % 300 == 0:
                logger.info(f"因子预载进度 {i + 1}/{len(pending_seed)} 天")

        # ── 2. 主回填：缺失日拉日K+因子，已完成日只补缺失的因子 ──
        trade_days = _trade_days(self.start_date, today)
        with sqlite3.connect(self.db_path) as conn:
            done_dates = {r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM stock_daily").fetchall()}
        todo = [d for d in trade_days if d not in done_dates]
        logger.info(
            f"主回填：{self.start_date} ~ {today} 共 {len(trade_days)} 个交易日，"
            f"已完成 {len(trade_days) - len(todo)} 天，待回填 {len(todo)} 天"
        )

        ok_days = fail_days = 0
        total_rows = 0
        for i, d in enumerate(trade_days):
            if d in done_dates:
                if not self._factor_day_done(d):
                    res = _with_retry(_fetch_factors, d, "因子补载")
                    if res is not None:
                        self._bulk_store_factors(res[0], res[1])
                        self._mark_factor_day(d)
                continue
            res = _with_retry(lambda day: self._bulk_fetch_day(bs, day), d, "批量拉取")
            if res is None:
                fail_days += 1
                continue
            k_rows, k_fields, f_rows, f_fields = res
            self._bulk_store_factors(f_rows, f_fields)
            self._mark_factor_day(d)
            recs = self._bulk_day_to_records(k_rows, k_fields, d)
            total_rows += self._bulk_write_day(d, recs)
            ok_days += 1
            if (i + 1) % 100 == 0:
                logger.info(
                    f"已处理 {i + 1}/{len(trade_days)} 天，成功 {ok_days} 失败 {fail_days}，累计 {total_rows} 行"
                )

        bs.logout()
        logger.info(f"批量回填完成 — 成功 {ok_days} 天 | 失败 {fail_days} 天 | 累计写入 {total_rows} 行")

    def _sync_today_bulk_api(self) -> int:
        """官方批量接口增量同步：只补最近缺失的交易日（2 次请求/天）。"""
        from datetime import date

        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            logger.warning(f"baostock 登录失败: {lg.error_msg}")
            return 0

        today = date.today().strftime("%Y-%m-%d")
        rs_cal = bs.query_trade_dates(start_date="2020-01-01", end_date=today)
        trade_days = []
        while rs_cal.next():
            row = rs_cal.get_row_data()
            if row[1] == "1":
                trade_days.append(row[0])

        with sqlite3.connect(self.db_path) as conn:
            done_dates = {r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM stock_daily").fetchall()}

        todo = [d for d in reversed(trade_days) if d not in done_dates][:3]
        if not todo:
            logger.info("所有股票已是最新，无需更新")
            bs.logout()
            return 0

        total = 0
        for d in reversed(todo):
            try:
                k_rows, k_fields, f_rows, f_fields = self._bulk_fetch_day(bs, d)
                self._bulk_store_factors(f_rows, f_fields)
                self._mark_factor_day(d)
                recs = self._bulk_day_to_records(k_rows, k_fields, d)
                total += self._bulk_write_day(d, recs)
                logger.info(f"批量同步 {d}：写入 {len(recs)} 条")
            except Exception as exc:
                logger.warning(f"批量同步 {d} 失败: {exc}")
        bs.logout()
        logger.info(f"sync_today_bulk（批量接口）: 共写入 {total} 条数据")
        return total

    def sync_today_bulk(self) -> int:
        """多进程并行通过 baostock 拉取增量数据（后复权），写入 SQLite。"""
        # 新版客户端 + 因子表就绪时走官方批量接口（2 次请求/天），否则走逐股多进程路径
        if self._bs_bulk_available() and self._has_factor_table():
            return self._sync_today_bulk_api()
        from datetime import date, timedelta
        from multiprocessing import Pool

        today_str = date.today().strftime("%Y-%m-%d")

        tasks = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
            ).fetchall()

        if not rows:
            logger.warning("本地无股票数据，请先执行 --backfill")
            return 0

        for symbol, last_date in rows:
            if last_date and last_date >= today_str:
                continue
            start = today_str
            if last_date:
                start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append((symbol, self._to_baostock_code(symbol), start, today_str))

        if not tasks:
            logger.info("所有股票已是最新，无需更新")
            return 0

        # 探测 baostock 可用性，失败则降级 akshare 增量同步
        import baostock as bs
        probe = bs.login()
        if probe.error_code != "0":
            logger.warning(f"baostock 登录失败: {probe.error_msg}，改用后备行情源增量同步")
            return self._sync_today_fallback(tasks)
        bs.logout()

        logger.info(f"需要更新 {len(tasks)} 只股票，启动多进程并行拉取...")

        n_workers = min(8, len(tasks))
        chunks = [tasks[i::n_workers] for i in range(n_workers)]

        with Pool(n_workers) as pool:
            batch_results = pool.map(_bs_fetch_batch, chunks)

        all_rows = []
        for batch in batch_results:
            all_rows.extend(batch)

        if not all_rows:
            logger.info("无新数据（可能非交易日）")
            return 0

        df = pd.DataFrame(all_rows, columns=["symbol", "date", "open", "high", "low", "close", "volume", "turnover", "turn_rate"])
        for col in ["open", "high", "low", "close", "volume", "turnover", "turn_rate"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["turn_rate"] = df["turn_rate"].fillna(0.0)
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]

        count = len(df)
        with sqlite3.connect(self.db_path) as conn:
            for d in df["date"].unique().tolist():
                conn.execute("DELETE FROM stock_daily WHERE date = ?", (d,))
            df.to_sql("stock_daily", conn, if_exists="append", index=False, method="multi", chunksize=500)
            conn.commit()

        logger.info(f"sync_today_bulk: 写入 {count} 条数据")
        return count

    def backfill(self, symbols: list[str]) -> None:
        """通过 baostock 批量回填历史日 K 线数据（后复权）。

        容错机制：
        - 单只股票失败自动重试 3 次，间隔递增（2s/4s/8s）
        - 每 200 只股票自动重连 baostock（防止长连接超时）
        - 已入库的自动 skip，中断后可重跑续传
        """
        # 新版客户端可用时走官方批量接口（~N天×2 次请求），否则走逐股路径
        if self._bs_bulk_available():
            self.backfill_bulk()
            return
        import time
        from datetime import date, timedelta

        import baostock as bs

        today_str = date.today().strftime("%Y-%m-%d")
        max_retries = 3
        reconnect_interval = 200  # 每处理 N 只股票重连一次

        def _login():
            lg = bs.login()
            if lg.error_code != "0":
                logger.error(f"baostock 登录失败: {lg.error_msg}")
                return False
            return True

        if not _login():
            logger.warning("baostock 登录失败，降级到后备行情源回填")
            self._backfill_fallback(symbols)
            return

        success = 0
        skipped = 0
        failed = 0
        since_reconnect = 0

        try:
            for i, symbol in enumerate(symbols):
                last_date = self._get_last_date(symbol)
                if last_date and last_date >= today_str:
                    skipped += 1
                    if (i + 1) % 500 == 0:
                        logger.info(
                            f"已处理 {i + 1}/{len(symbols)}，"
                            f"成功 {success} 跳过 {skipped} 失败 {failed}"
                        )
                    continue

                # 定期重连，防止长连接超时
                since_reconnect += 1
                if since_reconnect >= reconnect_interval:
                    bs.logout()
                    time.sleep(1)
                    if not _login():
                        logger.error("重连失败，终止回填")
                        return
                    since_reconnect = 0

                start = last_date or self.start_date
                if last_date:
                    start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

                bs_code = self._to_baostock_code(symbol)

                # 带重试的查询
                rows = []
                query_ok = False
                for attempt in range(max_retries):
                    try:
                        rs = bs.query_history_k_data_plus(
                            bs_code,
                            "date,open,high,low,close,volume,amount,turn",
                            start_date=start,
                            end_date=today_str,
                            frequency="d",
                            adjustflag="1",  # 后复权
                        )

                        if rs.error_code != "0":
                            raise RuntimeError(rs.error_msg)

                        rows = []
                        while rs.next():
                            rows.append(rs.get_row_data())
                        query_ok = True
                        break

                    except Exception as exc:
                        if attempt < max_retries - 1:
                            wait = 2 ** (attempt + 1)
                            logger.warning(
                                f"[{symbol}] 第{attempt + 1}次失败: {exc}，{wait}s 后重试"
                            )
                            time.sleep(wait)
                            # 重连 baostock
                            bs.logout()
                            time.sleep(1)
                            _login()
                        else:
                            logger.warning(f"[{symbol}] {max_retries}次重试均失败，跳过")

                if not query_ok:
                    failed += 1
                    continue

                if not rows:
                    skipped += 1
                    continue

                df = pd.DataFrame(rows, columns=rs.fields)
                for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["turn"] = df["turn"].fillna(0.0)
                df = df.dropna(subset=["close"])
                df = df[df["volume"] > 0]

                if df.empty:
                    skipped += 1
                    continue

                df["symbol"] = symbol
                df = df.rename(columns={"amount": "turnover", "turn": "turn_rate"})
                df = df[["symbol", "date", "open", "high", "low", "close", "volume", "turnover", "turn_rate"]]

                try:
                    with sqlite3.connect(self.db_path) as conn:
                        df.to_sql(
                            "stock_daily", conn, if_exists="append",
                            index=False, method="multi", chunksize=500,
                        )
                except sqlite3.IntegrityError:
                    pass

                success += 1

                if (i + 1) % 500 == 0:
                    logger.info(
                        f"已处理 {i + 1}/{len(symbols)}，"
                        f"成功 {success} 跳过 {skipped} 失败 {failed}"
                    )

        finally:
            bs.logout()

        logger.info(f"回填完成 — 成功: {success} | 跳过: {skipped} | 失败: {failed}")

    # ── 股票列表 ──

    def get_all_symbols(self) -> list[str]:
        """获取全市场 A 股代码列表（优先 baostock，失败自动降级 akshare）。"""
        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
        else:
            try:
                rs = bs.query_stock_basic(code_name="", code="")
                symbols = []
                while rs.next():
                    row = rs.get_row_data()
                    code = row[0]           # "sh.600000" or "sz.000001"
                    status = row[4]         # "1" = 上市
                    stock_type = row[5]     # "1" = 股票
                    if status == "1" and stock_type == "1":
                        symbols.append(code.split(".")[1])  # 提取纯数字代码
                logger.info(f"获取股票列表完成，共 {len(symbols)} 只")
                return symbols
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
            finally:
                bs.logout()

        # ── baostock 不可用：降级到 akshare ──
        logger.warning("baostock 不可用，改用 akshare 获取股票列表")
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            symbols = df["code"].astype(str).str.zfill(6).tolist()
            logger.info(f"获取股票列表完成（akshare），共 {len(symbols)} 只")
            return symbols
        except Exception as e:
            logger.error(f"akshare 获取股票列表失败: {e}")
            return []

    def get_local_symbols(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM stock_daily"
            ).fetchall()
        return [row[0] for row in rows]

    # ── 后备行情源（baostock 被封/不可用时自动降级：东财优先，腾讯备用）──

    @staticmethod
    def _em_fetch_range(symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """直接请求东方财富日K接口拉取后复权行情（含真实成交额），返回标准 schema。

        说明：不使用 akshare.stock_zh_a_hist —— 其硬编码的 ut 参数已失效，
        带 ut 的请求会被服务器直接断开连接（RemoteDisconnected）。

        Args:
            symbol: 6 位纯数字股票代码。
            start/end: 'YYYY-MM-DD' 格式日期范围。

        Returns:
            标准列（symbol/date/open/high/low/close/volume/turnover）的 DataFrame，
            失败返回 None。
        """
        import random
        import time

        import requests

        market_code = 1 if symbol.startswith("6") else 0
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            "klt": "101",                       # 日K
            "fqt": "2",                         # 2 = 后复权
            "secid": f"{market_code}.{symbol}",
            "beg": start.replace("-", ""),
            "end": end.replace("-", ""),
        }
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

        for attempt in range(3):
            try:
                time.sleep(random.uniform(0.3, 0.6))  # 随机间隔，降低反爬/限流风险
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                data = resp.json()
                if data.get("rc") != 0 or not data.get("data"):
                    return None
                klines = data["data"].get("klines")
                if not klines:
                    return None
                # f51..f58: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅
                out = pd.DataFrame(
                    [k.split(",") for k in klines],
                    columns=["date", "open", "close", "high", "low", "volume", "turnover", "amplitude"],
                )
                out["symbol"] = symbol
                out["volume"] = out["volume"].astype(float) * 100  # 东财单位为手，统一换算为股（与 baostock 一致）
                out = out[["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]]
                for col in ["open", "high", "low", "close", "volume", "turnover"]:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
                out = out.dropna(subset=["close"])
                out = out[out["volume"] > 0]
                return out
            except Exception as exc:
                if attempt < 2:
                    logger.warning(f"[{symbol}] 东财行情第{attempt + 1}次失败: {exc}，重试中")
                    time.sleep(2 ** (attempt + 1))
                else:
                    logger.warning(f"[{symbol}] 东财行情 3次重试均失败，跳过")
        return None

    @staticmethod
    def _tx_fetch_range(symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """请求腾讯财经日K接口拉取后复权行情（无成交额字段，用 收盘价×成交量 近似）。

        腾讯接口单次最多返回 640 根K线且 count 参数不能超过 640，
        因此按 500 天窗口分页拉取后合并。
        """
        import random
        import time
        from datetime import date as _date
        from datetime import timedelta as _td

        import requests

        prefix = "sh" if symbol.startswith(("6", "9")) else (
            "bj" if symbol.startswith(("4", "8")) else "sz"
        )
        # 腾讯行情接口有多个镜像主机，轮流使用可绕开单主机限流
        tx_hosts = [
            "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get",
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        ]
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://gu.qq.com/",
        }

        def fetch_window(w_start: str, w_end: str) -> pd.DataFrame | None:
            params = {"param": f"{prefix}{symbol},day,{w_start},{w_end},640,hfq"}
            for attempt in range(3):
                try:
                    time.sleep(random.uniform(0.3, 0.6))
                    url = tx_hosts[attempt % len(tx_hosts)]
                    resp = requests.get(url, params=params, headers=headers, timeout=15)
                    data = resp.json()
                    node = (data.get("data") or {}).get(f"{prefix}{symbol}") or {}
                    rows = node.get("hfqday") or node.get("qfqday") or node.get("day")
                    if not rows:
                        return None
                    # 分红除权日的行会追加第 7 个元素（分红信息 dict），统一截取前 6 个字段
                    rows = [r[:6] for r in rows]
                    return pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
                except Exception as exc:
                    if attempt < 2:
                        logger.warning(f"[{symbol}] 腾讯行情第{attempt + 1}次失败: {exc}，重试中")
                        time.sleep(2 ** (attempt + 1))
                    else:
                        logger.warning(f"[{symbol}] 腾讯行情 3次重试均失败，跳过")
            return None

        parts: list[pd.DataFrame] = []
        cur = _date.fromisoformat(start)
        end_d = _date.fromisoformat(end)
        window = _td(days=500)  # ≈340 个交易日，低于腾讯 640 根上限
        while cur <= end_d:
            w_end = min(cur + window, end_d)
            part = fetch_window(cur.isoformat(), w_end.isoformat())
            if part is not None and not part.empty:
                parts.append(part)
            cur = w_end + _td(days=1)

        if not parts:
            return None

        out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["date"])
        out["symbol"] = symbol
        out = out[["symbol", "date", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["close"])
        out = out[out["volume"] > 0]
        out["volume"] = out["volume"] * 100              # 手 → 股（与 baostock 一致）
        out["turnover"] = out["close"] * out["volume"]   # 成交额近似值
        return out

    def _probe_fallback_source(self) -> str:
        """探测可用后备行情源：优先东财（真实成交额），失败用腾讯（成交额近似）。

        Returns:
            'em' 或 'tx'；全部不可用时返回空字符串。
        """
        probe = self._em_fetch_range("000001", "2026-09-01", "2026-09-29")
        if probe is not None and not probe.empty:
            logger.info("后备行情源探测：东方财富可用（含真实成交额）")
            return "em"
        probe = self._tx_fetch_range("000001", "2026-09-01", "2026-09-29")
        if probe is not None and not probe.empty:
            logger.warning("东方财富不可用，改用腾讯行情源（成交额为 收盘价×成交量 近似值）")
            return "tx"
        logger.error("所有后备行情源均不可用，无法继续")
        return ""

    def _backfill_fallback(self, symbols: list[str]) -> None:
        """后备数据源回填：多线程拉取历史日K并写入 SQLite（中断后可续传）。

        运行开始时探测一次数据源（东财→腾讯）；运行中若连续大量失败
        （说明被限流），自动切换到另一数据源继续。
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import date, timedelta

        source = ""
        for attempt in range(20):
            source = self._probe_fallback_source()
            if source:
                break
            logger.warning(f"行情源全部被限流（第 {attempt + 1}/20 次探测），60 秒后重试...")
            time.sleep(60)
        if not source:
            logger.error("20 次探测均失败，终止回填；稍后重新运行 --backfill 可断点续传")
            return

        today_str = date.today().strftime("%Y-%m-%d")
        source_name = {"em": "东方财富", "tx": "腾讯"}[source]
        logger.info(f"后备回填开始：共 {len(symbols)} 只股票，数据源={source_name}，2 线程并行")

        state_lock = threading.Lock()
        state = {"source": source, "consecutive_fail": 0}

        def fetch_one(symbol: str) -> tuple[str, str]:
            last_date = self._get_last_date(symbol)
            if last_date and last_date >= today_str:
                return ("skip", symbol)
            start = last_date or self.start_date
            if last_date:
                start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

            with state_lock:
                src = state["source"]
            fetcher = self._em_fetch_range if src == "em" else self._tx_fetch_range
            out = fetcher(symbol, start, today_str)

            if out is None or out.empty:
                cooldown = False
                with state_lock:
                    state["consecutive_fail"] += 1
                    cf = state["consecutive_fail"]
                    if cf >= 20:
                        other = "tx" if state["source"] == "em" else "em"
                        state["source"] = other
                        state["consecutive_fail"] = 0
                        logger.warning(
                            "连续 20 次失败，切换数据源 → "
                            + ("腾讯" if other == "tx" else "东方财富")
                        )
                    elif cf % 5 == 0:
                        cooldown = True
                if cooldown:
                    logger.info("连续失败，冷却 30 秒以避免触发限流...")
                    time.sleep(30)
                return ("failed", symbol)

            with state_lock:
                state["consecutive_fail"] = 0
            with sqlite3.connect(self.db_path) as conn:
                try:
                    out.to_sql("stock_daily", conn, if_exists="append",
                               index=False, method="multi", chunksize=500)
                except sqlite3.IntegrityError:
                    pass
            return ("ok", symbol)

        success = skipped = failed = 0
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(fetch_one, s): s for s in symbols}
            done = 0
            for fut in as_completed(futures):
                status, _ = fut.result()
                if status == "ok":
                    success += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                done += 1
                if done % 500 == 0:
                    logger.info(
                        f"已处理 {done}/{len(symbols)}，成功 {success} 跳过 {skipped} 失败 {failed}"
                    )

        logger.info(f"回填完成（后备数据源） — 成功: {success} | 跳过: {skipped} | 失败: {failed}")

    def _sync_today_fallback(self, tasks: list) -> int:
        """后备数据源增量同步：多线程补拉最新数据并写入 SQLite。

        Args:
            tasks: [(symbol, bs_code, start, end), ...] 与 baostock 路径相同结构，
                   其中 bs_code 不被使用。
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        source = ""
        for attempt in range(10):
            source = self._probe_fallback_source()
            if source:
                break
            logger.warning(f"行情源全部被限流（第 {attempt + 1}/10 次探测），60 秒后重试...")
            time.sleep(60)
        if not source:
            logger.error("10 次探测均失败，本次增量同步终止")
            return 0

        source_name = {"em": "东方财富", "tx": "腾讯"}[source]
        logger.info(f"后备增量同步：{len(tasks)} 只股票，数据源={source_name}，4 线程并行")

        state_lock = threading.Lock()
        state = {"source": source, "consecutive_fail": 0}
        all_rows: list[pd.DataFrame] = []

        def fetch_one(task: tuple) -> None:
            symbol, _bs_code, start, end = task
            with state_lock:
                src = state["source"]
            fetcher = self._em_fetch_range if src == "em" else self._tx_fetch_range
            out = fetcher(symbol, start, end)
            if out is not None and not out.empty:
                with state_lock:
                    state["consecutive_fail"] = 0
                all_rows.append(out)
            else:
                with state_lock:
                    state["consecutive_fail"] += 1
                    if state["consecutive_fail"] >= 20:
                        other = "tx" if state["source"] == "em" else "em"
                        state["source"] = other
                        state["consecutive_fail"] = 0
                        logger.warning(
                            "连续 20 次失败，切换数据源 → "
                            + ("腾讯" if other == "tx" else "东方财富")
                        )

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(fetch_one, t) for t in tasks]
            for fut in as_completed(futures):
                fut.result()  # 拉取函数内部已捕获异常

        if not all_rows:
            logger.info("无新数据（可能非交易日）")
            return 0

        df = pd.concat(all_rows, ignore_index=True)
        count = len(df)
        with sqlite3.connect(self.db_path) as conn:
            for d in df["date"].unique().tolist():
                conn.execute("DELETE FROM stock_daily WHERE date = ?", (d,))
            df.to_sql("stock_daily", conn, if_exists="append",
                      index=False, method="multi", chunksize=500)
            conn.commit()

        logger.info(f"sync_today_bulk（后备数据源）: 写入 {count} 条数据")
        return count
