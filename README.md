# Sequoia-X · A股智能选股系统

> 15 个选股策略 + 机器学习「20 日大涨」概率模型 + baostock 批量数据引擎
>
> 每日收盘后自动：增量同步行情 → 运行全部策略 → 飞书推送 + 生成 Markdown 选股报告

---

## ✨ 特性

- **15 个选股策略**：覆盖趋势突破、动量（RPS）、形态识别、事件驱动、机器学习五大类
- **ML 大涨概率模型**：HistGradientBoosting 预测「未来 20 个交易日内涨幅 ≥15%」的概率，
  严格时间外推验证（AUC 0.66，P≥0.7 信号单笔均值 +3.28%、胜率 59.4%）
- **数据引擎**：baostock 0.9.4 每日批量接口（全市场一天 2 次请求），本地计算后复权，
  入库真实换手率与 isST 标记，全程断点续传
- **策略共振**：多策略命中同一股票时生成高/中吸引力组合卡片
- **每日 Markdown 报告**：含概览、各策略明细、共振板块与雪球链接
- **飞书推送**：策略专属机器人路由 + 指数退避重试

---

## 🚀 快速开始

```bash
# 1. 克隆
git clone https://github.com/james0608tt/Sequoia-X.git
cd Sequoia-X

# 2. 安装依赖（推荐 uv）
uv sync

# 3. 配置
cp .env.example .env
#   必填：FEISHU_WEBHOOK_URL（飞书机器人 webhook，兜底推送）
#   可选：STRATEGY_WEBHOOK_<策略标识>、REPORT_OUTPUT_DIR、baostock 限速参数

# 4. 首次回填历史数据（约 1~2 小时，断点续传可反复重跑）
python main.py --backfill

# 5. 日常运行（增量同步 + 15 策略 + 推送 + 报告）
python main.py
```

macOS 可双击 `launch_sequoia_x.command`（日常）或 `launch_sequoia_x_backfill.command`（回填）。

---

## 📊 15 个选股策略

### 趋势突破类

| 策略 | 逻辑 |
|---|---|
| **TurtleTrade 海龟交易** | 收盘突破前 20 日高点 + 成交额过亿 + 实体阳线且真涨，按流通市值降序 |
| **FirstNewHigh30DaysBreakout 30日首次新高** | 首次突破 30 日高点 + 均线多头发散 + 放量不爆量 + 成交额 >5 亿 |
| **RpsBreakout RPS 突破** | 120 日涨幅全市场排名前 10% + 贴近 120 日高点 |
| **RpsYilin 亦霖版 RPS 严格突破** | RPS ≥95 + 真突破 120 日高点 + 放量 + 趋势过滤，Top 30 |

### 均线与量能类

| 策略 | 逻辑 |
|---|---|
| **MaVolume 均量突破** | 5 日均线上穿 20 日均线 + 量 >20 日均量 1.5 倍 |
| **TripleVolumeBreakout 三倍量盘整突破** | 三倍量吸筹信号 → 缩量盘整 → 综合评分突破，Top 8 |

### 形态与超跌类

| 策略 | 逻辑 |
|---|---|
| **HighTightFlag 高紧旗形** | 40 日涨超 60% + 10 日振幅 <15% + 高位抗跌 + 缩量 |
| **LimitUpShakeout 涨停洗盘** | 昨日涨停 + 今日放量收阴但不破昨收 |
| **UptrendLimitDown 上升趋势跌停** | 多头趋势中的放量跌停级长阴（错杀修复）|
| **BowlRebound 碗口反弹** | 短期趋势线 > 多空线 + 15 日内放量阳线异动 + KDJ 超卖 + 价格回落碗中 |
| **TrendMeanReversion 趋势超跌回归** | 趋势健康 + 低于 MA20 3%~10% + 5 日回调 6%~18% + 止跌确认 |
| **StopFall 止跌候选** | 七因子打分 + 市场宽度动态分数线，S2 初步企稳 / S3 止跌确认分级 |
| **ThreeCycle 三周期（陈凯）** | 周线多头定性 + 日线旗形/圆弧/收敛图形 + 放量或新高切入点，换手率 5%~8% |

### 事件与机器学习类

| 策略 | 逻辑 |
|---|---|
| **PrivatePlacement 定增监控** | akshare 近 7 日定向增发公告（事件驱动）|
| **MlSurge ML 大涨概率** | 21 特征梯度提升模型预测「20 日内涨幅 ≥15%」概率，P≥0.7 输出 Top 50，自动剔除 ST |

---

## 🤖 ML 模型说明（第 15 个策略）

**目标**：预测未来 20 个交易日内最高涨幅 ≥15% 的概率。

- **算法**：HistGradientBoosting（sklearn），21 个技术特征（动量/均线偏离/量比/距高低点/ATR/RPS/换手率/成交额对数）
- **训练**：全市场 278 万样本，剔除 ST，严格按时间切分（训练 <2025-06，测试 ≥2026-01）
- **测试集成绩**：AUC 0.659；全局 Top1% 命中率 66.6%（基准 40.7%）

**执行规则回测**（2026-01 ~ 2026-09，冻结模型，T+1 开盘买入）：

| 方案 | 平均/笔 | 胜率 | 盈亏比 |
|---|---|---|---|
| P≥0.7 + 持有20天（推荐）| **+3.28%** | **59.4%** | 2.36 |
| 每日 Top5% + 持有20天 | +0.57% | 47.5% | 1.15 |
| P≥0.7 + 止损-10%/止盈+15% | -0.04% | 45.2% | 0.99 ⚠️ |

等权组合（P≥0.7，日均 189 个信号）：9 个月 +8.0%，最大回撤 -19.6%。

> ⚠️ 重要结论：**止损+止盈反而把期望打到 0**——这类股票常「先回撤再反弹」，止损会砍掉赢家。
> 测试期仅为 2026 年单一行情样本，务必组合化执行、分散持有。

**重新训练**（数据更新后建议定期重训）：

```bash
python scripts/train_surge_model.py      # 含ST版（参考）
python scripts/fetch_st_flags.py         # 采集每日 isST 标记（首次需）
python scripts/train_nost_backtest.py    # 非ST版 + 止损回测（正式模型）
python scripts/backtest_exec.py          # 执行方案矩阵回测
```

---

## 🗄️ 数据架构

- **数据源**：baostock 0.9.4 每日批量接口（`query_daily_history_k_AStock` + `query_daily_adjust_factor`）
  —— 全市场一天仅 **2 次请求**，从根上规避逐股查询的限流问题
- **复权**：本地按官方公式计算后复权（`不复权价 × backAdjustFactor`），与服务器后复权零差异
- **入库字段**：OHLCV、成交额、**真实换手率**、isST 标记
- **断点续传**：K线按交易日、因子按天标记，任何中断重跑即续
- **后备源**：baostock 不可用时自动降级东财/腾讯行情源

---

## ⚙️ 配置（.env）

```env
DB_PATH=data/sequoia_v2.db
START_DATE=2024-01-01
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx   # 必填
STRATEGY_WEBHOOK_MA_VOLUME=...     # 各策略专属机器人（可选，未配置走默认）

# baostock 限速保护
BAOSTOCK_MAX_WORKERS=2
BAOSTOCK_REQUEST_DELAY_SECONDS=0.8
BAOSTOCK_REQUEST_JITTER_SECONDS=0.8
BAOSTOCK_ERROR_COOLDOWN_SECONDS=30

# 报告输出目录（空=默认 ~/Documents/量化交易/今日选股结果）
REPORT_OUTPUT_DIR=
```

---

## 🛠️ scripts 工具集

| 脚本 | 用途 |
|---|---|
| `train_surge_model.py` | 特征构建 + 含ST版 ML 训练 + 全市场扫描 |
| `train_nost_backtest.py` | 非ST版 ML 训练 + 三种止损线回测 |
| `backtest_exec.py` | 执行方案矩阵回测（入场×止损×止盈×信号）|
| `fetch_st_flags.py` | 采集每日 isST 标记（断点续传）|
| `quick_select.py` | 跳过同步，直接对现有库跑全部策略 + 报告 |
| `analyze_stock.py` | 单只股票 14 策略全项体检（后复权口径）|
| `analyze_raw.py` | 单只股票体检（不复权口径）|
| `bench.py` | 策略耗时基准 |

---

## 🧪 测试

```bash
uv run pytest   # 19 个测试
```

---

## ⚠️ 免责声明

本项目仅供量化研究与学习交流，所有选股信号不构成任何投资建议。股市有风险，入市需谨慎。
