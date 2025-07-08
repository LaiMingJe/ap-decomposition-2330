# src/visualize.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 設定專業的視覺樣式
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'

# 專業投行配色方案 (Bank of America inspired)
COLORS = {
    'primary_red': '#E31837',      # BOA 主紅色
    'deep_burgundy': '#730237',    # 深酒紅色
    'navy_blue': '#022873',        # 深藍色
    'sky_blue': '#0487D9',         # 天藍色
    'light_gray': '#EDEDED',       # 淺灰色
    'positive': '#27AE60',         # 成功綠（保留）
    'negative': '#E67E22',         # 警告橙（保留）
    'accent': '#F39C12',           # 強調黃（保留）
    'passive': '#022873',          # 被動策略色
    'active': '#E31837'            # 主動策略色
}

def create_ap_focused_analysis(df_passive, df_active, passive_metrics, active_metrics, ap_results):
    """
    生成完整的 AP 分解學術研究圖表集
    
    設計理念：
    - 學術專業性：清晰的圖表標題和標注
    - 視覺層次感：重點突出，次要信息適度
    - 備審友善性：圖表獨立完整，便於選用
    """
    
    import os
    os.makedirs('results/ap_analysis', exist_ok=True)
    
    print("📊 生成學術研究圖表...")
    
    # 設定非互動模式
    plt.ioff()
    
    try:
        # 圖表1：策略績效比較 (核心圖表)
        print("   生成圖表1: 策略績效比較")
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        create_performance_comparison(ax1, df_passive, df_active, passive_metrics, active_metrics)
        plt.tight_layout()
        plt.savefig('results/ap_analysis/01_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # 圖表2：AP分解核心結果
        print("   生成圖表2: AP分解理論驗證")
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 7))
        create_ap_decomposition_chart(ax2a, ap_results)
        create_theory_validation_table(ax2b, ap_results, passive_metrics, active_metrics)
        plt.tight_layout()
        plt.savefig('results/ap_analysis/02_ap_decomposition_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # 圖表3：權重-報酬關係與主動貢獻
        print("   生成圖表3: 權重報酬關係分析")
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7))
        create_weight_return_analysis(ax3a, df_active)
        create_active_component_timeline(ax3b, df_passive, df_active)
        plt.tight_layout()
        plt.savefig('results/ap_analysis/03_weight_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # 圖表4：超額報酬分析
        print("   生成圖表4: 超額報酬特徵分析")
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 7))
        create_excess_return_distribution(ax4a, df_passive, df_active)
        create_rolling_performance(ax4b, df_passive, df_active)
        plt.tight_layout()
        plt.savefig('results/ap_analysis/04_excess_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)

        # 圖表5：策略績效綜合比較表
        print("   生成圖表5: 策略績效綜合比較")
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 8))
        create_performance_metrics_table(ax5a, passive_metrics, active_metrics)
        create_rolling_ap_analysis(ax5b, df_active)
        plt.tight_layout()
        plt.savefig('results/ap_analysis/05_comprehensive_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)

        print("✅ 所有圖表已生成並保存至 results/ap_analysis/")
        
    except Exception as e:
        print(f"❌ 圖表生成錯誤: {e}")

def create_performance_comparison(ax, df_passive, df_active, passive_metrics, active_metrics):
    """策略績效比較圖表 - 專業投行風格"""
    
    # 主要曲線 - 使用BOA配色
    ax.plot(df_passive['Date'], df_passive['NAV'],
           label='被動策略 (Passive DCA)', linewidth=3, 
           color=COLORS['navy_blue'], alpha=0.9)
    ax.plot(df_active['Date'], df_active['NAV'],
           label='主動策略 (Momentum DCA)', linewidth=3, 
           color=COLORS['primary_red'], alpha=0.9)

    # 超額報酬填充區域
    ax.fill_between(df_active['Date'], df_passive['NAV'], df_active['NAV'],
                    where=(df_active['NAV'] >= df_passive['NAV']),
                    alpha=0.25, color=COLORS['positive'], label='正向超額報酬')
    ax.fill_between(df_active['Date'], df_passive['NAV'], df_active['NAV'],
                    where=(df_active['NAV'] < df_passive['NAV']),
                    alpha=0.25, color=COLORS['negative'], label='負向超額報酬')

    # 績效摘要資訊框
    excess_return = active_metrics['total_return'] - passive_metrics['total_return']
    
    info_text = f"""績效摘要:
被動策略: {passive_metrics['total_return']:.1%}
主動策略: {active_metrics['total_return']:.1%}
超額報酬: {excess_return:.1%}
夏普比率改善: {active_metrics['sharpe_ratio'] - passive_metrics['sharpe_ratio']:.3f}"""
    
    ax.text(0.65, 0.25, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
           facecolor=COLORS['light_gray'], edgecolor=COLORS['deep_burgundy'], alpha=0.95))

    ax.set_title('投資策略績效比較\nStrategy Performance Comparison', 
                fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('時間 (Time)', fontsize=12)
    ax.set_ylabel('組合淨值 (Portfolio NAV)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=COLORS['light_gray'])
    
    # BOA投行風格座標軸設計
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['deep_burgundy'])
    ax.spines['bottom'].set_color(COLORS['deep_burgundy'])

def create_ap_decomposition_chart(ax, ap_results):
    """AP分解結果柱狀圖 - Lo (2007) 理論實現"""
    
    components = ['Active (δp)', 'Passive (νp)']
    values = [ap_results['Active (δp)'], ap_results['Passive (νp)']]
    
    # 主動與被動成分配色方案
    colors = [COLORS['primary_red'], COLORS['sky_blue']]
    
    # 創建柱狀圖
    bars = ax.bar(components, values, color=colors, alpha=0.8, 
                 edgecolor=COLORS['navy_blue'], linewidth=2, width=0.6)

    # 添加數值標籤
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., 
               height + 0.0001 * np.sign(height),
               f'{value:.6f}', ha='center',
               va='bottom' if height > 0 else 'top',
               fontweight='bold', fontsize=11, color=COLORS['navy_blue'])

    # 突出顯示主動比率 - BOA風格
    theta_p = ap_results['Active Ratio (θp)']
    ax.text(0.5, 0.85, f'Active Ratio (θp) = {theta_p:.4f}',
            transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['accent'], 
                     alpha=0.8, edgecolor=COLORS['navy_blue']))
    
    # 添加理論解釋
    interpretation = "正值表示主動擇時創造價值" if theta_p > 0 else "需檢討擇時策略"
    ax.text(0.5, 0.75, interpretation, transform=ax.transAxes, 
           ha='center', fontsize=10, style='italic', color=COLORS['deep_burgundy'])

    ax.set_title('Lo (2007) AP Decomposition Results\nAP分解核心結果', 
                fontweight='bold', fontsize=13, pad=15)
    ax.set_ylabel('Component Value', fontsize=11)
    ax.axhline(y=0, color=COLORS['navy_blue'], linestyle='--', alpha=0.7, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y', color=COLORS['light_gray'])
    
    # BOA風格的座標軸
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['deep_burgundy'])
    ax.spines['bottom'].set_color(COLORS['deep_burgundy'])

def create_theory_validation_table(ax, ap_results, passive_metrics, active_metrics):
    """理論驗證結果表格"""
    ax.axis('off')

    delta_p = ap_results['Active (δp)']
    theta_p = ap_results['Active Ratio (θp)']
    correlation = ap_results.get('Weight-Return Correlation', 0)
    excess_return = active_metrics['total_return'] - passive_metrics['total_return']

    # 理論驗證結果
    validations = [
        ['驗證項目', '數值結果', '理論符合度'],
        ['主動成分 (δp)', f'{delta_p:.6f}', 'PASS' if delta_p > 0 else 'FAIL'],
        ['主動比率 (θp)', f'{theta_p:.4f}', 'PASS' if 0 < theta_p < 1 else 'WARN'],
        ['權重-報酬相關性', f'{correlation:.3f}', 'PASS' if abs(correlation) > 0.1 else 'FAIL'],
        ['策略超額報酬', f'{excess_return:.2%}', 'PASS' if excess_return > 0 else 'FAIL'],
        ['整體理論一致性', 
         '高度符合' if delta_p > 0 and 0 < theta_p < 1 else '部分符合', 
         'PASS' if delta_p > 0 else 'WARN']
    ]

    # 創建表格
    table = ax.table(cellText=validations, cellLoc='center', loc='center',
                     colWidths=[0.45, 0.3, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # BOA風格美化表格
    for i in range(len(validations)):
        for j in range(len(validations[0])):
            cell = table[(i, j)]
            if i == 0:  # 標題行 - 使用BOA深藍色
                cell.set_facecolor(COLORS['navy_blue'])
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.12)
            else:
                if j == 2:  # 結果列
                    if 'PASS' in validations[i][j]:
                        cell.set_facecolor('#D5EFDB')  # 淺綠
                        cell.set_text_props(color='green', weight='bold')
                    elif 'WARN' in validations[i][j]:
                        cell.set_facecolor('#FFF2CC')  # 淺黃
                        cell.set_text_props(color='orange', weight='bold')
                    elif 'FAIL' in validations[i][j]:
                        cell.set_facecolor('#FADBD8')  # 淺紅
                        cell.set_text_props(color='red', weight='bold')
                else:
                    cell.set_facecolor(COLORS['light_gray'])
                cell.set_height(0.08)
            
            cell.set_edgecolor(COLORS['deep_burgundy'])
            cell.set_linewidth(1)

    ax.set_title('Lo (2007) 理論驗證結果\nTheory Validation Results', 
                fontsize=13, fontweight='bold', pad=20)

def create_weight_return_analysis(ax, df_active):
    """權重-報酬關係散布圖 - 增強統計資訊"""
    
    if 'Weight' not in df_active.columns or 'Return' not in df_active.columns:
        ax.text(0.5, 0.5, '權重資料不可用\nWeight data unavailable', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        return

    weights = df_active['Weight'].dropna()
    returns = df_active['Return'].dropna()
    min_len = min(len(weights), len(returns))
    weights = weights.iloc[-min_len:]
    returns = returns.iloc[-min_len:] * 100  # 轉為百分比

    # 散布圖
    scatter = ax.scatter(weights, returns, alpha=0.6, s=25, c=returns, 
                        cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
    
    # 趨勢線和統計
    if len(weights) > 10:
        z = np.polyfit(weights, returns, 1)
        p = np.poly1d(z)
        ax.plot(weights, p(weights), "r-", alpha=0.8, linewidth=2.5)

        correlation = np.corrcoef(weights, returns)[0, 1]
        r_squared = correlation ** 2
        
        # 統計資訊框
        stats_text = f'相關係數: {correlation:.3f}\nR²: {r_squared:.3f}\n觀察值: {len(weights)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', 
                        alpha=0.8, edgecolor='navy'))

    ax.set_title('權重-報酬關係分析 (AP分解理論核心)\nWeight-Return Relationship', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('動態權重 (Dynamic Weight)', fontsize=11)
    ax.set_ylabel('日報酬率 (%)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加顏色條
    plt.colorbar(scatter, ax=ax, label='報酬率 (%)', shrink=0.8)

def create_active_component_timeline(ax, df_passive, df_active):
    """累積主動貢獻時間序列"""
    
    active = df_active['NAV'].pct_change().dropna()
    passive = df_passive['NAV'].pct_change().dropna()
    min_len = min(len(active), len(passive))
    excess = (active.iloc[-min_len:] - passive.iloc[-min_len:]) * 100
    cumulative_excess = excess.cumsum()

    # 主線圖
    ax.plot(df_active['Date'].iloc[-min_len:], cumulative_excess,
            color=COLORS['positive'], linewidth=3, alpha=0.9)
    
    # 填充區域
    ax.fill_between(df_active['Date'].iloc[-min_len:], 0, cumulative_excess,
                    alpha=0.3, color=COLORS['positive'],
                    where=(cumulative_excess >= 0), interpolate=True)
    ax.fill_between(df_active['Date'].iloc[-min_len:], 0, cumulative_excess,
                    alpha=0.3, color=COLORS['negative'],
                    where=(cumulative_excess < 0), interpolate=True)

    # 添加統計資訊
    final_excess = cumulative_excess.iloc[-1]
    max_excess = cumulative_excess.max()
    min_excess = cumulative_excess.min()
    
    stats_text = f'最終累積: {final_excess:.2f}%\n最高峰: {max_excess:.2f}%\n最低谷: {min_excess:.2f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                    alpha=0.9, edgecolor='gray'))

    ax.set_title('累積主動貢獻軌跡\nCumulative Active Contribution', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('累積超額報酬 (%)', fontsize=11)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)

def create_excess_return_distribution(ax, df_passive, df_active):
    """超額報酬分佈直方圖與統計分析"""
    
    active = df_active['NAV'].pct_change().dropna()
    passive = df_passive['NAV'].pct_change().dropna()
    min_len = min(len(active), len(passive))
    excess = (active.iloc[-min_len:] - passive.iloc[-min_len:]) * 100
    
    # 超額報酬分佈直方圖
    n, bins, patches = ax.hist(excess, bins=35, alpha=0.7, color=COLORS['primary_red'], 
                              edgecolor='black', linewidth=0.5, density=True)
    
    # 顏色漸變
    for i, p in enumerate(patches):
        if bins[i] < 0:
            p.set_facecolor(COLORS['negative'])
        else:
            p.set_facecolor(COLORS['positive'])
        p.set_alpha(0.7)

    # 統計線
    mean_excess = excess.mean()
    std_excess = excess.std()
    
    ax.axvline(mean_excess, color='red', linestyle='--', linewidth=2.5, 
              label=f'平均值: {mean_excess:.3f}%')
    ax.axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=2, 
              label='零超額報酬')
    
    # 添加正態分佈曲線
    x = np.linspace(excess.min(), excess.max(), 100)
    normal_curve = (1/(std_excess * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_excess)/std_excess)**2)
    ax.plot(x, normal_curve, 'navy', linewidth=2, alpha=0.8, label='理論常態分佈')

    # 統計摘要
    skewness = excess.skew()
    kurtosis = excess.kurtosis()
    win_rate = (excess > 0).mean()
    
    stats_text = f'''統計特徵:
勝率: {win_rate:.1%}
偏度: {skewness:.3f}
峰度: {kurtosis:.3f}
標準差: {std_excess:.3f}%'''
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', 
                    alpha=0.9, edgecolor='orange'))

    ax.set_title('日超額報酬分佈特徵\nDaily Excess Return Distribution', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('日超額報酬 (%)', fontsize=11)
    ax.set_ylabel('密度 (Density)', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

def create_rolling_performance(ax, df_passive, df_active):
    """30天滾動績效比較"""
    
    passive_rolling = df_passive['NAV'].pct_change().rolling(30).mean() * 100
    active_rolling = df_active['NAV'].pct_change().rolling(30).mean() * 100

    ax.plot(df_passive['Date'], passive_rolling, label='被動策略', 
           color=COLORS['passive'], linewidth=2, alpha=0.8)
    ax.plot(df_active['Date'], active_rolling, label='主動策略', 
           color=COLORS['primary_red'], linewidth=2, alpha=0.8)
    
    # 填充優勢區域
    ax.fill_between(df_active['Date'], passive_rolling, active_rolling,
                    where=(active_rolling >= passive_rolling),
                    alpha=0.2, color=COLORS['positive'], interpolate=True)
    ax.fill_between(df_active['Date'], passive_rolling, active_rolling,
                    where=(active_rolling < passive_rolling),
                    alpha=0.2, color=COLORS['negative'], interpolate=True)

    ax.set_title('30天滾動報酬比較\n30-Day Rolling Return Comparison', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('滾動報酬率 (%)', fontsize=11)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)

def create_rolling_ap_analysis(ax, df_active):
    """滾動AP分解分析 - 增強統計指標"""
    try:
        from src.ap_decomposition import rolling_ap_decomposition
        rolling = rolling_ap_decomposition(df_active, window=60)
        
        if not rolling.empty and len(rolling) > 0:
            # 主線圖
            ax.plot(rolling['Date'], rolling['Active_Ratio'], 
                   linewidth=2.5, color=COLORS['accent'], alpha=0.9, 
                   label='滾動主動比率')
            ax.fill_between(rolling['Date'], 0, rolling['Active_Ratio'], 
                           alpha=0.3, color=COLORS['accent'])
            
            # 統計線
            mean_ratio = rolling['Active_Ratio'].mean()
            median_ratio = rolling['Active_Ratio'].median()
            std_ratio = rolling['Active_Ratio'].std()
            
            ax.axhline(y=mean_ratio, color='red', linestyle=':', linewidth=2,
                      label=f'平均值: {mean_ratio:.3f}')
            ax.axhline(y=median_ratio, color='purple', linestyle='--', linewidth=1.5,
                      label=f'中位數: {median_ratio:.3f}')
            
            # 統計指標計算
            # 穩定性與趨勢分析
            positive_periods = (rolling['Active_Ratio'] > 0).mean()
            volatility = rolling['Active_Ratio'].std()
            trend = 'improving' if rolling['Active_Ratio'].iloc[-5:].mean() > rolling['Active_Ratio'].iloc[:5].mean() else 'declining'
            
            stats_text = f'''滾動AP統計摘要:
平均主動比率: {mean_ratio:.4f}
標準差: {std_ratio:.4f}
正值比例: {positive_periods:.1%}
近期趨勢: {trend}
觀察窗口: 60天'''
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.4", 
                            facecolor=COLORS['light_gray'], 
                            alpha=0.95, edgecolor=COLORS['deep_burgundy']))
            
            ax.set_title('滾動AP分解分析 (60天窗口)\nRolling AP Decomposition Analysis', 
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_ylabel('主動比率 (θp)', fontsize=11)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '滾動分析資料不足\nInsufficient data for rolling analysis', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('滾動AP分解分析', fontsize=13, fontweight='bold')
    except Exception as e:
        ax.text(0.5, 0.5, f'分析錯誤: {str(e)}', transform=ax.transAxes, 
               ha='center', va='center', fontsize=10)
        ax.set_title('滾動AP分解分析', fontsize=13, fontweight='bold')

def create_performance_metrics_table(ax, passive_metrics, active_metrics):
    """策略績效比較表格"""
    ax.axis('off')
    
    # 指標定義
    metrics_info = [
        ('總報酬率', 'total_return', '%'),
        ('年化報酬率', 'annualized_return', '%'), 
        ('年化波動率', 'annualized_volatility', '%'),
        ('夏普比率', 'sharpe_ratio', ''),
        ('最大回撤', 'max_drawdown', '%'),
        ('勝率', 'win_rate', '%'),
        ('卡爾馬比率', 'calmar_ratio', ''),
        ('索丁諾比率', 'sortino_ratio', '')
    ]
    
    # 準備表格資料
    table_data = []
    for name, key, unit in metrics_info:
        if unit == '%':
            passive_val = f"{passive_metrics[key]:.2%}"
            active_val = f"{active_metrics[key]:.2%}"
        else:
            passive_val = f"{passive_metrics[key]:.3f}"
            active_val = f"{active_metrics[key]:.3f}"
        
        table_data.append([name, passive_val, active_val])
    
    # 創建績效比較表格
    table = ax.table(cellText=table_data, 
                    colLabels=['績效指標', '被動策略', '主動策略'], 
                    cellLoc='center', loc='center',
                    colWidths=[0.5, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 表格美化與績效著色
    for i in range(len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # 表頭
                cell.set_facecolor(COLORS['navy_blue'])
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.15)
            else:
                # 主動策略績效著色
                if j == 2:
                    try:
                        metric_key = metrics_info[i-1][1]
                        if metric_key in ['total_return', 'annualized_return', 'sharpe_ratio', 
                                        'win_rate', 'calmar_ratio', 'sortino_ratio']:
                            # 數值越大越好的指標
                            if active_metrics[metric_key] > passive_metrics[metric_key]:
                                cell.set_facecolor('#D5F5E3')  # 綠色表示優勢
                            else:
                                cell.set_facecolor('#FADBD8')  # 紅色表示劣勢
                        elif metric_key in ['annualized_volatility', 'max_drawdown']:
                            # 數值越小越好的指標
                            if abs(active_metrics[metric_key]) < abs(passive_metrics[metric_key]):
                                cell.set_facecolor('#D5F5E3')  # 綠色表示優勢
                            else:
                                cell.set_facecolor('#FADBD8')  # 紅色表示劣勢
                    except:
                        cell.set_facecolor('#F8F9FA')
                else:
                    cell.set_facecolor('#F8F9FA')
                
                cell.set_height(0.1)
            
            cell.set_edgecolor(COLORS['deep_burgundy'])
            cell.set_linewidth(1)

    ax.set_title('策略績效比較表\nPerformance Metrics Comparison', 
                fontsize=13, fontweight='bold', pad=20)
    
    # 表格說明
    note_text = "綠色表示該策略在此指標上表現更佳"
    ax.text(0.5, 0.02, note_text, transform=ax.transAxes, 
           ha='center', fontsize=9, style='italic', color='gray')