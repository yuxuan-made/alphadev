import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= 配置区域 =================
START_DATE = '2024-01-01'
END_DATE = '2024-03-31'
# 基础路径 (会自动在其下创建 symbol/1m/...)
BASE_DIR = './data/futures/um/daily/klines'
# ===========================================

def get_robust_session():
    session = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=1, 
                  status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

session = get_robust_session()

def get_all_usdt_symbols():
    print("正在从币安接口获取全量币种列表...")
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = [s["symbol"] for s in data["symbols"] 
                   if s.get("contractType") == "PERPETUAL" 
                   and s.get("quoteAsset") == "USDT" 
                   and s.get("status") == "TRADING"]
        print(f"成功获取！当前共有 {len(symbols)} 个可交易合约。")
        return symbols
    except Exception as e:
        print(f"获取列表失败: {e}")
        return []

def download_and_process(symbol, date_str):
    # 构造新的文件路径结构：klines/symbol/1m/symbol-1m-date.parquet
    save_dir = os.path.join(BASE_DIR, symbol, '1m')
    file_name = f"{symbol}-1m-{date_str}.parquet"
    save_path = os.path.join(save_dir, file_name)
    
    # 如果文件已存在，这里简单跳过 (如果你想覆盖旧格式，可以注释掉这两行)
    if os.path.exists(save_path):
        return "EXISTS"

    url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{date_str}.zip"
    
    try:
        response = session.get(url, timeout=(5, 20))
        if response.status_code == 404: return "NOT_FOUND"
        if response.status_code != 200: return "ERROR"

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None, low_memory=False)

                # 智能跳过表头行 (如果存在)
                first_cell = str(df.iloc[0, 0])
                if "open" in first_cell.lower() or not first_cell.replace('.', '', 1).isdigit():
                    df = df.iloc[1:]

                # === 修改点：保留所有12列 ===
                # 原始定义: 0:OpenTime, 1:Open, 2:High, 3:Low, 4:Close, 5:Vol, 6:CloseTime, 
                # 7:QuoteVol, 8:Count, 9:TakerBuyVol, 10:TakerBuyQuoteVol, 11:Ignore
                df.columns = [
                    'open_time',            # 0. 开盘时间
                    'open',                 # 1. 开盘价
                    'high',                 # 2. 最高价
                    'low',                  # 3. 最低价
                    'close',                # 4. 收盘价
                    'volume',               # 5. 成交量
                    'close_time',           # 6. 收盘时间 (保留)
                    'quote_volume',         # 7. 成交额
                    'count',                # 8. 笔数
                    'taker_buy_volume',     # 9. 主动买入量
                    'taker_buy_quote_volume',# 10. 主动买入额
                    'ignore'                # 11. Ignore (保留)
                ]
                
                # 强制转数字
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df.dropna(inplace=True)
                if df.empty: return "EMPTY"

                # 时间戳转换 (两个时间都转)
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                # 保存
                os.makedirs(save_dir, exist_ok=True)
                df.to_parquet(save_path, index=False)
                return "SUCCESS"

    except Exception as e:
        if "404" not in str(e):
            print(f"\nError on {symbol} {date_str}: {str(e)}")
        return "ERROR"

def main():
    symbols = get_all_usdt_symbols()
    if not symbols: return

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    date_range = [start + timedelta(days=x) for x in range((end - start).days + 1)]

    total_tasks = len(symbols) * len(date_range)
    pbar = tqdm(total=total_tasks, desc="Total Progress", unit="file")

    for symbol in symbols:
        for date_obj in date_range:
            date_str = date_obj.strftime("%Y-%m-%d")
            status = download_and_process(symbol, date_str)
            pbar.update(1)
            if status == "SUCCESS":
                time.sleep(random.uniform(0.1, 0.2))
                
    pbar.close()
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()