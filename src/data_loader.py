import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import Optional

def get_yfinance_data(
    ticker: str,
    start: str = '2019-01-01', 
    end: Optional[str] = None,
    interval: str = '1d',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    從 Yahoo Finance 取得股票或 ETF 歷史資料。

    此函式為整個投資策略回測之資料來源，確保資料品質與一致性，
    並支援多種金融商品與時間頻率，為後續量化分析提供穩定的數據基礎。

    參數:
    - ticker: 股票代號 (如 '2330.TW'=台積電, 'AAPL'=蘋果)
    - start: 開始日期 (yyyy-mm-dd)
    - end: 結束日期 (yyyy-mm-dd)，預設為今日
    - interval: 資料頻率 ('1d'=日線, '1wk'=週線, '1mo'=月線)
    - save_path: 資料儲存路徑

    回傳:
    - DataFrame: 標準化 OHLCV 表格
    """

    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    print(f"[Info] 開始下載 {ticker} 歷史資料：{start} ~ {end} (頻率：{interval})")

    try:
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            progress=True
        )

        if df.empty:
            raise ValueError(f"[Error] 無法取得 {ticker} 的資料，請確認代號正確性。")

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        required_cols = ['Date', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"[Error] 資料缺少必要欄位：{missing_cols}")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)

        print(f"[Info] 已成功取得 {ticker}：共 {len(df)} 筆資料")
        print(f"[Info] 期間：{df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"[Info] 價格範圍：${df['Close'].min():.2f} ~ ${df['Close'].max():.2f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"[Info] 資料已儲存至：{save_path}")

        return df

    except Exception as e:
        print(f"[Error] 下載 {ticker} 資料時發生例外：{e}")
        raise


def get_multiple_tickers_data(
    tickers: list,
    start: str = '2019-01-01',
    end: Optional[str] = None,
    interval: str = '1d',
    save_dir: str = 'results'
) -> dict:
    """
    批量下載多檔股票資料，用於投資組合分析與比較。

    參數:
    - tickers: 股票代號清單
    - start, end, interval: 與 get_yfinance_data() 相同
    - save_dir: 儲存目錄

    回傳:
    - dict: {ticker: DataFrame}
    """
    results = {}
    successful_downloads = 0

    print(f"[Info] 開始批次下載，總共 {len(tickers)} 檔標的。")

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[Info] ({i}/{len(tickers)}) 處理中：{ticker}")
            save_path = os.path.join(save_dir, f"{ticker.replace('.', '_')}.csv")
            df = get_yfinance_data(ticker, start, end, interval, save_path)
            results[ticker] = df
            successful_downloads += 1
        except Exception as e:
            print(f"[Warning] 跳過 {ticker}：{e}")
            continue

    print(f"[Info] 批次下載完成：成功 {successful_downloads}/{len(tickers)} 檔 ({successful_downloads/len(tickers)*100:.1f}%)")
    return results


def analyze_data_quality(df: pd.DataFrame, ticker: str) -> dict:
    """
    資料品質檢查：統計缺值、異常值、價格分佈等。

    主要檢核：
    - 缺失值
    - 異常日內波動
    - 資料連續性
    """
    analysis = {
        'ticker': ticker,
        'total_rows': len(df),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'missing_values': df.isnull().sum().to_dict(),
        'price_stats': {
            'min': df['Close'].min(),
            'max': df['Close'].max(),
            'mean': df['Close'].mean(),
            'std': df['Close'].std()
        }
    }

    daily_returns = df['Close'].pct_change().dropna()
    extreme_moves = daily_returns[abs(daily_returns) > 0.2]
    analysis['extreme_moves'] = len(extreme_moves)

    return analysis