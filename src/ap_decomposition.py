import pandas as pd
import numpy as np
from typing import Dict, Tuple

def compute_ap_decomposition(df: pd.DataFrame) -> Dict[str, float]:
    """
    Lo (2007) Active-Passive Decomposition 核心實現

    理論基礎：
    Lo, Andrew W. (2007) "The Statistics of Sharpe Ratios"
    Financial Analysts Journal, Vol. 63, No. 4, pp. 36-52

    數學框架：
    投資組合期望報酬可分解為兩個正交成分：
    E[R_p] = δp + νp

    其中：
    1. Active Component (δp) = Cov(w_t, r_t)
       - 權重與報酬的共變異數
       - 衡量動態擇時 (market timing) 所帶來的 α 價值
       - 正值表示主動擇時有效，負值表示擇時失效

    2. Passive Component (νp) = E[w_t] × E[r_t]
       - 平均權重乘以平均報酬
       - 衡量市場曝險 (β) 的貢獻
       - 反映被動持有策略的基礎報酬

    3. Active Ratio (θp) = δp / (δp + νp)
       - 主動成分在總報酬中的相對比重
       - 範圍 [-1, 1]，正值表示擇時成功，值越接近 1 表示主動比重越高

    學術意義：
    此分解框架提供客觀量化主動管理效果的工具，
    超越僅比較總報酬的限制，是現代投資組合
    績效歸因與管理的基礎之一。

    本模組建立於作者於中央研究院擔任研究助理期間之理論學習與匯報成果，
    展現將學術理論轉化為可執行程式的完整鏈條。
    """

    if 'Weight' not in df.columns or 'Return' not in df.columns:
        raise ValueError("AP 分解需要包含 Weight 與 Return 欄位")

    df_clean = df[['Weight', 'Return']].dropna()

    if len(df_clean) < 10:
        return {
            'Active (δp)': 0.0,
            'Passive (νp)': 0.0,
            'Active Ratio (θp)': 0.0,
            'Data Quality': 'Insufficient data for reliable decomposition',
            'Sample Size': len(df_clean)
        }

    w = df_clean['Weight'].values
    r = df_clean['Return'].values

    try:
        # === 核心 AP 分解計算 ===

        # 1. 主動成分：權重與報酬的共變異數
        cov_matrix = np.cov(w, r, ddof=1)
        delta_p = cov_matrix[0, 1]  # Cov(w_t, r_t)

        # 2. 被動成分：平均權重 × 平均報酬
        nu_p = np.mean(w) * np.mean(r)

        # 3. 主動比率
        total = delta_p + nu_p
        theta_p = delta_p / total if abs(total) > 1e-10 else 0

        # === 統計診斷 ===
        correlation = np.corrcoef(w, r)[0, 1]
        n_obs = len(df_clean)
        statistical_significance = 'High' if n_obs > 100 else 'Moderate' if n_obs > 50 else 'Low'

        results = {
            'Active (δp)': delta_p,
            'Passive (νp)': nu_p,
            'Active Ratio (θp)': theta_p,
            'Weight-Return Correlation': correlation,
            'Sample Size': n_obs,
            'Statistical Significance': statistical_significance,
            'Weight Mean': np.mean(w),
            'Weight Std': np.std(w),
            'Return Mean': np.mean(r),
            'Return Std': np.std(r)
        }

        return results

    except Exception as e:
        print(f"[Warning] AP 分解計算錯誤: {e}")
        return {
            'Active (δp)': 0.0,
            'Passive (νp)': 0.0,
            'Active Ratio (θp)': 0.0,
            'Error': str(e)
        }


def analyze_ap_components(df_active: pd.DataFrame, df_passive: pd.DataFrame) -> Dict[str, float]:
    """
    AP 擴充分析：比較主動策略與被動策略之超額表現與風險

    提供多面向的時間序列指標：
    - 年化超額報酬
    - 主動風險
    - 資訊比率
    - 勝率、回撤、偏度、峰度等

    此分析有助於理解主動成分如何在不同市場狀況下發揮效果。
    """

    active_returns = df_active['NAV'].pct_change().dropna()
    passive_returns = df_passive['NAV'].pct_change().dropna()

    min_length = min(len(active_returns), len(passive_returns))
    active_returns = active_returns.iloc[-min_length:]
    passive_returns = passive_returns.iloc[-min_length:]

    excess_returns = active_returns - passive_returns

    analysis = {
        'active_contribution': excess_returns.mean() * 252,
        'active_volatility': excess_returns.std() * np.sqrt(252),
        'information_ratio': (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0,
        'positive_periods_ratio': (excess_returns > 0).mean(),
        'maximum_active_drawdown': calculate_active_drawdown(excess_returns),
        'excess_return_skewness': excess_returns.skew(),
        'excess_return_kurtosis': excess_returns.kurtosis()
    }

    if 'Weight' in df_active.columns:
        weights = df_active['Weight'].dropna()

        analysis.update({
            'average_weight': weights.mean(),
            'weight_volatility': weights.std(),
            'weight_range': (weights.min(), weights.max()),
            'extreme_weight_frequency': ((weights > 1.2) | (weights < 0.8)).mean(),
            'weight_turnover': weights.diff().abs().mean()
        })

    return analysis


def calculate_active_drawdown(excess_returns: pd.Series) -> float:
    """
    計算主動管理超額報酬之最大回撤
    """
    cumulative_excess = (1 + excess_returns).cumprod()
    running_max = cumulative_excess.expanding().max()
    drawdown = (cumulative_excess - running_max) / running_max
    return drawdown.min()


def validate_ap_theory(ap_results: Dict[str, float]) -> Dict[str, str]:
    """
    檢驗 AP 分解結果是否符合理論預期。

    核心依據：
    - δp 正值表示主動擇時成立
    - θp 在 0~1 合理區間
    - 權重-報酬關係應具統計相關性
    - 樣本量需足以保證穩定性
    """
    validation = {}

    delta_p = ap_results.get('Active (δp)', 0)
    theta_p = ap_results.get('Active Ratio (θp)', 0)
    correlation = ap_results.get('Weight-Return Correlation', 0)
    sample_size = ap_results.get('Sample Size', 0)

    if delta_p > 0.001:
        validation['Active Component'] = f"✅ 正向主動價值：δp = {delta_p:.6f}，擇時有效，符合理論"
    elif delta_p < -0.001:
        validation['Active Component'] = f"❌ 負向主動價值：δp = {delta_p:.6f}，擇時失效"
    else:
        validation['Active Component'] = f"⚪ 中性主動價值：δp ≈ 0"

    if 0 < theta_p < 1:
        validation['Active Ratio'] = f"✅ θp = {theta_p:.4f}，{theta_p:.1%} 由主動擇時貢獻"
    elif theta_p >= 1:
        validation['Active Ratio'] = f"⚠️ θp = {theta_p:.4f}，異常偏高"
    elif theta_p <= 0:
        validation['Active Ratio'] = f"❌ θp = {theta_p:.4f}，主動策略未產生價值"

    if abs(correlation) > 0.1:
        correlation_strength = "強" if abs(correlation) > 0.3 else "中等" if abs(correlation) > 0.2 else "弱"
        correlation_direction = "正" if correlation > 0 else "負"
        validation['Weight-Return Relationship'] = f"✅ {correlation_strength}{correlation_direction}相關 (r={correlation:.3f})"
    else:
        validation['Weight-Return Relationship'] = f"❌ 相關性微弱 (r={correlation:.3f})"

    if sample_size > 100:
        validation['Statistical Reliability'] = f"✅ 高可靠性 (N={sample_size})"
    elif sample_size > 50:
        validation['Statistical Reliability'] = f"⚠️ 中等可靠性 (N={sample_size})"
    else:
        validation['Statistical Reliability'] = f"❌ 低可靠性 (N={sample_size})"

    return validation


def rolling_ap_decomposition(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    滾動 AP 分解：觀察 δp 與 θp 隨時間之穩定性
    """
    if 'Weight' not in df.columns or 'Return' not in df.columns:
        return pd.DataFrame()

    rolling_results = []

    for i in range(window, len(df)):
        window_data = df.iloc[i-window:i]
        ap_result = compute_ap_decomposition(window_data)

        rolling_results.append({
            'Date': df.iloc[i]['Date'] if 'Date' in df.columns else i,
            'Active_Component': ap_result.get('Active (δp)', 0),
            'Passive_Component': ap_result.get('Passive (νp)', 0),
            'Active_Ratio': ap_result.get('Active Ratio (θp)', 0),
            'Weight_Return_Correlation': ap_result.get('Weight-Return Correlation', 0)
        })

    return pd.DataFrame(rolling_results)


def compare_ap_strategies(strategies_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    多策略 AP 分解結果比較
    """
    comparison_results = []

    for strategy_name, strategy_df in strategies_dict.items():
        try:
            ap_result = compute_ap_decomposition(strategy_df)

            comparison_results.append({
                'Strategy': strategy_name,
                'Active_Component': ap_result.get('Active (δp)', 0),
                'Passive_Component': ap_result.get('Passive (νp)', 0),
                'Active_Ratio': ap_result.get('Active Ratio (θp)', 0),
                'Weight_Return_Correlation': ap_result.get('Weight-Return Correlation', 0),
                'Sample_Size': ap_result.get('Sample Size', 0)
            })

        except Exception as e:
            print(f"[Warning] 策略 {strategy_name} AP 分解失敗: {e}")
            continue

    comparison_df = pd.DataFrame(comparison_results)

    if len(comparison_df) > 1:
        comparison_df['Active_Rank'] = comparison_df['Active_Component'].rank(ascending=False)
        comparison_df['Active_Ratio_Rank'] = comparison_df['Active_Ratio'].rank(ascending=False)

    return comparison_df


def generate_ap_research_report(ap_results: Dict[str, float],
                                ap_analysis: Dict[str, float],
                                validation: Dict[str, str]) -> str:
    """
    生成 AP 分解研究報告
    """
    report = f"""
# Active-Passive Decomposition 實證研究報告

## 理論框架
- 主動成分 (δp): 共變異數 Cov(w_t, r_t) = {ap_results.get('Active (δp)', 0):.6f}
- 被動成分 (νp): E[w_t] × E[r_t] = {ap_results.get('Passive (νp)', 0):.6f}
- 主動比率 (θp): δp/(δp+νp) = {ap_results.get('Active Ratio (θp)', 0):.4f}

## 驗證結果
- {validation.get('Active Component')}
- {validation.get('Active Ratio')}
- {validation.get('Weight-Return Relationship')}
- {validation.get('Statistical Reliability')}

## 擴充指標
- 年化主動貢獻: {ap_analysis.get('active_contribution', 0):.4f}
- 資訊比率: {ap_analysis.get('information_ratio', 0):.4f}
- 勝率: {ap_analysis.get('positive_periods_ratio', 0):.2%}
"""

    return report


def save_ap_research_documentation(ap_results: Dict[str, float],
                                   ap_analysis: Dict[str, float]) -> None:
    """
    保存 AP 分解研究結果
    """
    validation = validate_ap_theory(ap_results)
    report = generate_ap_research_report(ap_results, ap_analysis, validation)

    import os
    os.makedirs('results/ap_analysis', exist_ok=True)

    with open('results/ap_analysis/ap_research_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    import json
    detailed_results = {
        'ap_decomposition_results': ap_results,
        'extended_analysis': ap_analysis,
        'theory_validation': validation
    }

    with open('results/ap_analysis/detailed_ap_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)

    print("[Info] AP 分解研究報告已保存")