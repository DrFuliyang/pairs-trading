# Pairs Trading Strategy

> Cointegration-based mean reversion strategy for AI coding assistants. Drop into `~/.claude/skills/pairs-trading/SKILL.md` and start building.

**Features:** Cointegration screening · Z-score signals · Backtest engine · Window optimization · A股/港股/美股适配

---

## Quick Start

```bash
mkdir -p ~/.claude/skills/pairs-trading
curl -sL https://raw.githubusercontent.com/GARCHSigma/pairs-trading/refs/heads/master/SKILL.md \
  -o ~/.claude/skills/pairs-trading/SKILL.md
```

Then ask your AI coding assistant: *"帮我找港股两只高股息的配对，要求协整"* — it will activate the strategy automatically.

---

## Strategy Logic

```
Z-score > +1  → Short ratio (sell S1, buy S2)
Z-score < -1  → Long ratio  (buy S1, sell S2)
|Z-score| < 0.75 → Exit (close positions)
```

**核心假设：** 价差围绕均值回归。协整是必要条件——不协整的配对迟早会发散。

---

## Core Functions

| Function | Description |
|----------|-------------|
| `find_cointegrated_pairs(data)` | 遍历所有标的，输出协整对（p < 0.05） |
| `trade(S1, S2, window1, window2)` | 回测引擎，返回 profit |
| `zscore(series)` | Z-score 标准化 |
| `stationarity_test(X)` | ADF 平稳性检验 |

---

## Data Sources

| Source | Coverage | Auth |
|--------|----------|------|
| **Twelve Data** (default) | 美股/港股/数字货币 | Free tier: 500 req/day |
| **yfinance** | 美股 | Free |
| **akshare** | A股 | Free |
| **腾讯行情 (qt.gtimg.cn)** | 港股 78 字段 | Zero auth |

---

## Demo Output (Twelve Data, 252 days)

```
AAPL: $201.36 → $308.82 | ADF p=0.86 (非平稳)
MSFT: $454.86 → $418.57 | ADF p=0.79 (非平稳)
Cointegration p=0.83 → 不协整（2024-2025 美股分化期）

Ratio Z-score: [-1.65, +1.94]
|Z| > 1 的交易日: 114/252 (45% 有信号)
```

> 配对需要**定期重新筛选**。2024-2025 AAPL 单边上涨、MSFT 盘整，协整关系被打破。

---

## Window Optimization

Search 0–254 day windows to find the optimal rolling window:

```python
scores = [trade(S1, S2, l, 5) for l in range(255)]
best = np.argmax(scores)
# → Best train window: X days → profit: Y
```

---

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `window_fast` | 5 | 快线窗口 |
| `window_slow` | 60 | 慢线窗口（均值 + 标准差基准） |
| `entry_threshold` | ±1 σ | Z-score 入场阈值 |
| `exit_threshold` | 0.75 σ | 平仓阈值 |
| `coint_pvalue` | < 0.05 | 协整显著性 |

---

## A股适配

```python
import akshare as ak

def fetch_aio_closes(codes, start, end):
    dfs = []
    for code in codes:
        df = ak.stock_zh_a_hist(symbol=code, period='daily',
                                start_date=start, end_date=end)
        dfs.append(df[['日期','收盘']].rename(columns={'日期':'date'}).set_index('date'))
    return pd.concat(dfs, axis=1)

# 银行/保险同板块更易协整
codes = ['601398', '601288', '601988', '600036', '600000']
df = fetch_aio_closes(codes, '20200101', '20240101')
scores, pvalues, pairs = find_cointegrated_pairs(df)
```

> A股注意：涨跌停限制可能阻止价差回归；停牌会导致数据断裂。

---

## License

Apache 2.0