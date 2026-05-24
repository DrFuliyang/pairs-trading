---
name: pairs-trading
description: 配对交易策略 — 协整检验、配对筛选、Z-score 信号、窗口优化全流程。数据源用 yfinance（替换已废弃的 pandas_datareader）。用于 A 股/港股/美股均值回归策略开发。
origin: https://github.com/KidQuant/Pairs-Trading-With-Python
version: 1.0
---

# Pairs Trading Strategy (配对交易)

配对交易是均值回归策略的核心：找到协整的两个标的，当价差偏离均值时做多/做空价差，等待回归时平仓获利。

---

## 核心概念

### Stationarity（平稳性）
ADF 检验 `H_0`: 存在单位根（非平稳）。p-value < 0.01 才认为序列平稳。
```python
from statsmodels.tsa.stattools import adfuller

def stationarity_test(X, cutoff=0.01):
    pvalue = adfuller(X)[1]
    status = "STATIONARY" if pvalue < cutoff else "NON-STATIONARY"
    print(f"{X.name}: p={pvalue:.4f} → {status}")
    return pvalue < cutoff
```

### Cointegration（协整）vs Correlation（相关）
- **相关**：两个序列同向移动
- **协整**：两个序列的线性组合是平稳的（价差会回归）

关键例子：高相关不一定协整（如两个随机游走），低相关也可能协整。
```python
from statsmodels.tsa.stattools import coint

score, pvalue, _ = coint(X, Y)
# pvalue < 0.05 → 协整
```

---

## 全流程

### Step 1: 协整配对筛选

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import yfinance as yf

def find_cointegrated_pairs(data):
    """
    遍历所有标的，找协整对
    data: DataFrame (columns = tickers, index = date, values = close)
    return: score_matrix, pvalue_matrix, pairs
    """
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            try:
                score, pvalue, _ = coint(S1, S2)
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairs.append((keys[i], keys[j]))
            except Exception:
                pass

    return score_matrix, pvalue_matrix, pairs

# 数据获取（替换废弃的 pandas_datareader）
def fetch_closes(tickers, start='2013-01-01', end='2019-01-01'):
    df = yf.download(tickers, start=start, end=end, progress=False)['Close']
    return df

# 示例：科技股配对
tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']
df = fetch_closes(tickers)
scores, pvalues, pairs = find_cointegrated_pairs(df)
print(f"协整对: {pairs}")
```

### Step 2: 热力图可视化

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
mask = pvalues >= 0.05
sns.heatmap(pvalues, xticklabels=df.columns, yticklabels=df.columns,
            cmap='RdYlGn_r', mask=mask, annot=True, fmt='.3f', ax=ax)
plt.title("Cointegration p-value Heatmap (p < 0.05 = green)")
plt.show()
```

### Step 3: 计算 Spread / Ratio

两种方法选一：
```python
import statsmodels.api as sm

# 方法A: OLS 回归残差（Spread）
S1 = df['ADBE']
S2 = df['MSFT']
S1_const = sm.add_constant(S1)
results = sm.OLS(S2, S1_const).fit()
b = results.params['ADBE']
spread = S2 - b * S1  # 价差

# 方法B: 直接价比（Ratio）
ratio = S1 / S2
```

### Step 4: Z-Score 信号

```python
def zscore(series):
    return (series - series.mean()) / np.std(series)

# 画 Z-score 图
z = zscore(ratio)
z.plot(figsize=(12, 6))
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--', label='+1σ sell')
plt.axhline(-1.0, color='green', linestyle='--', label='-1σ buy')
plt.legend()
plt.show()
```

### Step 5: 信号生成（移动窗口）

```python
# 5日均线 vs 60日均线
window_fast = 5
window_slow = 60

ratios = df['ADBE'] / df['MSFT']
ma_fast = ratios.rolling(window=window_fast, center=False).mean()
ma_slow = ratios.rolling(window=window_slow, center=False).mean()
std = ratios.rolling(window=window_slow, center=False).std()
zscore_60_5 = (ma_fast - ma_slow) / std

# 可视化
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
ratios.plot(ax=axes[0])
ma_fast.plot(ax=axes[0], label='5d MA')
ma_slow.plot(ax=axes[0], label='60d MA')
axes[0].legend()
axes[0].set_title("ADBE/MSFT Ratio")

zscore_60_5.plot(ax=axes[1])
axes[1].axhline(0, color='black')
axes[1].axhline(1.0, color='red', linestyle='--')
axes[1].axhline(-1.0, color='green', linestyle='--')
axes[1].legend(['Z-Score', 'Mean', '+1', '-1'])
plt.show()
```

### Step 6: 回测交易

```python
def trade(S1, S2, window1, window2, exit_threshold=0.75):
    """
    配对交易策略
    
    window1: 快线窗口（均值）
    window2: 慢线窗口（基准 + 标准差）
    exit_threshold: 平仓 Z-score 阈值（< 0.75 时平仓）
    
    z > 1 → Short ratio（卖 S1 买 S2）
    z < -1 → Long ratio（买 S1 卖 S2）
    |z| < exit_threshold → 平仓
    """
    if window1 == 0 or window2 == 0:
        return 0

    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1, center=False).mean()
    ma2 = ratios.rolling(window=window2, center=False).mean()
    std = ratios.rolling(window=window2, center=False).std()
    z = (ma1 - ma2) / std

    money = 0
    countS1 = 0
    countS2 = 0

    for i in range(len(ratios)):
        if np.isnan(z.iloc[i]):
            continue
        if z.iloc[i] < -1:  # Long ratio
            money -= S1.iloc[i] - S2.iloc[i] * ratios.iloc[i]
            countS1 += 1
            countS2 -= ratios.iloc[i]
        elif z.iloc[i] > 1:  # Short ratio
            money += S1.iloc[i] - S2.iloc[i] * ratios.iloc[i]
            countS1 -= 1
            countS2 += ratios.iloc[i]
        elif abs(z.iloc[i]) < exit_threshold:  # Exit
            money += S1.iloc[i] * countS1 + S2.iloc[i] * countS2
            countS1 = 0
            countS2 = 0

    return money

# Train / Test split
train = ratios[:1057]
test = ratios[1057:]

# 在测试集上运行
score = trade(df['ADBE'].iloc[881:], df['MSFT'].iloc[881:], 60, 5)
print(f"Test profit: {score:.2f}")
```

### Step 7: 最优窗口搜索

```python
# 搜索最佳窗口（0-254）
length_scores = [
    trade(df['ADBE'].iloc[:1057], df['MSFT'].iloc[:1057], l, 5)
    for l in range(255)
]
best_length = np.argmax(length_scores)
print(f"Best window: {best_length} days → train score: {length_scores[best_length]:.2f}")

# 在测试集验证
test_scores = [
    trade(df['ADBE'].iloc[1057:], df['MSFT'].iloc[1057:], l, 5)
    for l in range(255)
]
best_test = np.argmax(test_scores)
print(f"Best test window: {best_test} days → test score: {test_scores[best_test]:.2f}")

# 可视化
plt.figure(figsize=(15, 7))
plt.plot(length_scores, label='Train')
plt.plot(test_scores, label='Test')
plt.xlabel("Window length")
plt.ylabel("Profit")
plt.legend()
plt.show()
```

---

## A 股适配

A股配对需注意：
- **ETF 优先**：50ETF(510050)、300ETF(510300)、创业板ETF(159915)，流动性好、无停牌
- **行业配对**：银行/保险/券商同板块更易协整
- **数据源**：`akshare` 或 `baostock` 替代 yfinance
- **涨跌停限制**：极端价差可能无法回归（需预留安全边际）

```python
import akshare as ak

# A股日线数据
def fetch_aio_closes(codes, start, end):
    dfs = []
    for code in codes:
        try:
            df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start, end_date=end)
            dfs.append(df[['日期', '收盘']].rename(columns={'日期': 'date', '收盘': code}).set_index('date'))
        except:
            pass
    return pd.concat(dfs, axis=1)
```

---

## 参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_fast` | 5 | 快线窗口（短均线） |
| `window_slow` | 60 | 慢线窗口（长均线+标准差） |
| `entry_threshold` | ±1 σ | 入场 Z-score 阈值 |
| `exit_threshold` | 0.75 σ | 平仓 Z-score 阈值 |
| `coint_pvalue` | < 0.05 | 协整检验显著性 |

---

## 信号逻辑总结

```
Z-score > +1  → Short ratio (空价差) = 卖 S1 + 买 S2
Z-score < -1  → Long ratio  (多价差) = 买 S1 + 卖 S2
|Z-score| < 0.75 → Exit (平仓)
```

核心假设：价差围绕均值回归。协整是必要条件——不协整的配对迟早会发散。