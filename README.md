# 📈 AP 分解理論實證研究系統 (Lo 2007)

## 🎓 研究背景

本專案依據 **Lo, Andrew W. (2007) _The Statistics of Sharpe Ratios_** 所提出之  
**Active-Passive Decomposition (AP 分解)** 理論，  
將投資組合的期望報酬嚴謹拆解為主動擇時成分 (α) 與市場曝險成分 (β)。

Lo (2007) Proposition 1 公式如下：

> **E[Rₚ] = Σ₍ᵢ₌₁₎ⁿ E[ωᵢt Rᵢt]**  
>   = Σ₍ᵢ₌₁₎ⁿ { Cov[ωᵢt, Rᵢt] + E[ωᵢt] E[Rᵢt] }  
>   = δp + νp

其中，

- **δp = Σ Cov[ωᵢt, Rᵢt]** → Active Component （投資組合權重與資產報酬之**共變異數成分**）
- **νp = Σ E[ωᵢt] E[Rᵢt]** → Passive Component （純市場曝險報酬）
- **θp = δp / (δp + νp)** → Active Ratio （主動成分在總報酬中的占比）

---

此理論指出，若投資組合權重與資產報酬之間存在正向**共變異數**（Cov > 0），  
則 δp 為正，表示投資組合中含有主動擇時報酬 (α)；  
若權重配置為靜態，則僅存在 νp，等同於單純承擔市場風險 (β)。

---

本專案以台積電 (2330.TW) 為實證標的，透過比較 **純被動定期定額 (Passive DCA)**  
與 **動量加權定期定額 (Momentum DCA)** 之表現，  
檢證實際投資策略中是否能觀察到 δp > 0 且 θp 顯著大於零，  
以量化主動擇時成分對總報酬之影響。

---

## 🎯 研究目的

- 驗證 Lo (2007) AP 分解理論於 **台股 (台積電 2330.TW)** 動量加權定期定額策略之適用性
- 比較 **純被動定期定額 (DCA)** 與 **動量調整後 DCA** 之績效差異
- 量化主動擇時效果 (α) 是否成立，並以視覺化與統計指標輔助呈現

---

## ⚙️ 執行環境與使用方式

### ✅ 執行環境

- Python ≥ 3.8
- 主要套件：
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - yfinance

---

### ✅ 執行步驟

```bash
# 1️⃣ 建立虛擬環境
python -m venv .venv

# 2️⃣ 啟用虛擬環境
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3️⃣ 安裝依賴套件
pip install -r requirements.txt

# 4️⃣ 執行主程式
python main.py

```
---

執行後將自動：

- 從 Yahoo Finance 下載台積電歷史股價

- 計算被動與動量策略的 NAV

- 執行 AP 分解，計算 δp、νp、θp

- 產生包含績效指標、報酬分佈、滾動分析等圖表

- 輸出 Markdown 與 JSON 研究報告於 /results/ap_analysis/

---

## 📁 專案結構
```bash
ap_decomposition_project/
 ├── main.py
 ├── src/
 │   ├── data_loader.py
 │   ├── strategies.py
 │   ├── ap_decomposition.py
 │   ├── backtest.py
 │   ├── visualize.py
 ├── results/
 ├── docs/
 │   └── research_summary.md
 ├── requirements.txt
 ├── README.md
 ├── .gitignore

```
---

## 📊 主要成果
✅ AP 分解結果：主動成分 δp 為正，主動比率 θp 約 44%，證明動態權重確實產生 α

✅ 績效指標：動量策略長期報酬顯著高於被動 DCA，夏普比率提高，最大回撤降低

✅ 視覺化輸出：含淨值曲線、AP 分解柱狀圖、權重-報酬關係、超額報酬分佈、滾動分析等 5 張核心圖

> 研究結果已另以 Markdown 與 JSON 形式保存，可供審查或簡報附檔使用。

---

## 📌 作者資訊
- 📚 **Lai Ming-Je (賴名哲)**  
- 🎓 **輔仁大學金融與國際企業學系**  
- 📬 **研究關鍵字：** AP 分解、主動擇時 α、量化投資、資產配置