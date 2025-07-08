import pandas as pd
import numpy as np

def calculate_performance_metrics(df, risk_free_rate=0.02):
    """
    投資組合績效指標計算工具

    提供標準化計算方法，衡量回測期間之報酬、波動度、風險調整指標，
    並支援 AP 分解研究及多策略比較之需求。

    計算內容：
    - 總報酬率、年化報酬率
    - 年化波動度、夏普比率
    - 最大回撤、Calmar、Sortino
    - VaR(95%) 為 5% 分位數之日報酬損失
    - 最大連續虧損期、勝率
    """

    if 'NAV' not in df.columns or 'Cumulative_Capital' not in df.columns:
        raise ValueError("缺少必要欄位：NAV 與 Cumulative_Capital")

    nav_returns = df['NAV'].pct_change().fillna(0)

    # === 基礎報酬指標 ===
    total_return = (df['NAV'].iloc[-1] / df['Cumulative_Capital'].iloc[-1]) - 1
    days = len(df)
    years = days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # === 風險指標 ===
    annualized_volatility = nav_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

    # === 回撤分析 ===
    cumulative_returns = (1 + nav_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # === 其他關鍵指標 ===
    win_rate = (nav_returns > 0).mean()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    negative_returns = nav_returns[nav_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

    var_95 = nav_returns.quantile(0.05)

    consecutive_losses = 0
    max_consecutive_losses = 0
    for ret in nav_returns:
        if ret < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    # 回傳核心績效指標
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'var_95': var_95,
        'max_consecutive_losses': max_consecutive_losses
    }