import pandas as pd
import numpy as np

def compute_daily_returns(df):
    """計算日報酬率"""
    df = df.copy()
    df['Return'] = df['Close'].pct_change().fillna(0)
    return df


def compute_dca_nav(df, monthly_amount=1.0):
    """
    定期定額 Buy-and-Hold 策略

    核心邏輯：
    - 每月第一個交易日投入固定金額
    - 按當日收盤價買入股票
    - 持股部位持續累積，不做調整

    本策略為 AP 分解的「純被動」基準組，對應市場曝險 (β)。
    """
    df = df.copy()

    if 'Date' not in df.columns:
        df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])

    df['Contribution'] = 0.0
    df['MonthYear'] = df['Date'].dt.to_period('M')
    first_day_mask = df.groupby('MonthYear')['Date'].transform('min') == df['Date']
    df.loc[first_day_mask, 'Contribution'] = monthly_amount

    df['Cumulative_Capital'] = df['Contribution'].cumsum()
    df = compute_daily_returns(df)

    units = 0.0
    nav = []

    for _, row in df.iterrows():
        contribution = row['Contribution']
        close_price = row['Close']

        if isinstance(contribution, pd.Series):
            contribution = contribution.iloc[0]
        if isinstance(close_price, pd.Series):
            close_price = close_price.iloc[0]

        if contribution > 0 and close_price > 0:
            buy_units = contribution / close_price
            units += buy_units

        total_value = units * close_price if close_price > 0 else 0
        nav.append(total_value)

    df['NAV'] = nav
    return df


def compute_momentum_dca_nav(df, lookback=5, monthly_amount=1.0, weight_config=None):
    """
    動量調整定期定額策略 (Momentum DCA)

    理論依據：
    依據 Lo (2007) AP 分解架構設計，藉由權重調整產生主動成分 (δp)，
    權重與報酬的共變異數 (Cov) 為主動擇時 α 的來源。

    創新特點：
    - 維持定期定額基本配置，兼具風險分散
    - 基於動量信號進行動態權重調整
    - 提供 AP 分解共變異數 δp 的實證樣本

    參數:
    - lookback: 動量指標回顧期，預設 5 日
    - weight_config: 權重調整參數，包含四階段門檻及調整倍率

    權重邏輯：
    - 過去 N 日累積報酬 > +5%：強勢加碼 (1.3x)
    - 0 ~ +5%：溫和加碼 (1.1x)
    - 0 ~ -5%：溫和減碼 (0.9x)
    - < -5%：大幅減碼 (0.7x)

    AP 解讀：
    成功時，將觀察到：
    - δp > 0 (共變異數為正)
    - 權重與報酬具顯著正相關
    - θp 在 0~1 合理區間
    """
    df = df.copy()

    if weight_config is None:
        weight_config = {
            'strong_up': 1.3,
            'mild_up': 1.1,
            'mild_down': 0.9,
            'strong_down': 0.7,
            'threshold': 0.05
        }

    if 'Date' not in df.columns:
        df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])

    df = compute_daily_returns(df)

    df['Rolling_Return'] = df['Close'].pct_change(periods=lookback).fillna(0)

    def calculate_weight(rolling_return, config):
        """
        根據動量信號決定權重：
        正報酬加碼，負報酬減碼，依強弱調整倍率。
        """
        if pd.isna(rolling_return):
            return 1.0
        elif rolling_return > config['threshold']:
            return config['strong_up']
        elif rolling_return > 0:
            return config['mild_up']
        elif rolling_return > -config['threshold']:
            return config['mild_down']
        else:
            return config['strong_down']

    df['Weight'] = df['Rolling_Return'].apply(lambda x: calculate_weight(x, weight_config))

    df['Contribution'] = 0.0
    df['MonthYear'] = df['Date'].dt.to_period('M')
    first_day_mask = df.groupby('MonthYear')['Date'].transform('min') == df['Date']
    df.loc[first_day_mask, 'Contribution'] = monthly_amount

    df['Cumulative_Capital'] = df['Contribution'].cumsum()

    nav = []
    prev_nav = 0.0

    for _, row in df.iterrows():
        c = row['Contribution']  # 當日資金流入
        r = row['Return']        # 當日報酬
        w = row['Weight']        # 當日權重

        if isinstance(c, pd.Series): c = c.iloc[0]
        if isinstance(r, pd.Series): r = r.iloc[0]
        if isinstance(w, pd.Series): w = w.iloc[0]

        # NAV 運算式：
        # NAV_t = [NAV_{t-1} + 當日投入] * (1 + 當日報酬 * 當日權重)
        prev_nav += c
        if not pd.isna(r) and not pd.isna(w):
            prev_nav *= (1 + r * w)

        nav.append(max(prev_nav, 0))

    df['NAV'] = nav
    return df