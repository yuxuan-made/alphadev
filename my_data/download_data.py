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
# 建议先用少量的币测试，没问题再换成完整的200个币的列表
SYMBOLS = ['ETHUSDT', 'BTCUSDT', 'ALPACAUSDT', '1000LUNCUSDT', 'SOLUSDT', 'PIPPINUSDT', 'ZECUSDT', 'LUNA2USDT', 'XRPUSDT', 'BNXUSDT', 'ALPHAUSDT', 'DOGEUSDT', 'BNBUSDT', 'SUIUSDT', 'PIEVERSEUSDT', 'FARTCOINUSDT', 'BCHUSDT', 'HYPEUSDT', 'PORT3USDT', 'POWERUSDT', 'USTCUSDT', '1000PEPEUSDT', 'LINKUSDT', 'BEATUSDT', 'ASTERUSDT', 'ACEUSDT', 'PUFFERUSDT', 'AVAXUSDT', 'ADAUSDT', 'ENAUSDT', 'PUMPUSDT', 'UXLINKUSDT', 'VIDTUSDT', 'SXPUSDT', 'AGIXUSDT', 'RLSUSDT', 'XNYUSDT', 'APTUSDT', 'BOBUSDT', 'SAPIENUSDT', 'LTCUSDT', 'LINAUSDT', 'MEMEFIUSDT', 'LEVERUSDT', 'NEIROETHUSDT', 'FTMUSDT', 'MONUSDT', 'XPLUSDT', 'GRIFFAINUSDT', 'WAVESUSDT', 'TAOUSDT', 'OMNIUSDT', 'AMBUSDT', 'AI16ZUSDT', 'BSWUSDT', 'OCEANUSDT', 'AAVEUSDT', 'UNIUSDT', 'PENGUUSDT', 'SWARMSUSDT', 'DOTUSDT', 'TRADOORUSDT', '1000SHIBUSDT', 'NEARUSDT', 'TRXUSDT', 'WIFUSDT', 'GIGGLEUSDT', 'STRAXUSDT', 'THEUSDT', 'FILUSDT', 'WLDUSDT', 'RENUSDT', 'UNFIUSDT', 'TIAUSDT', 'LIGHTUSDT', 'YBUSDT', 'TRUMPUSDT', 'TNSRUSDT', 'DGBUSDT', 'EGLDUSDT', 'TURBOUSDT', 'TROYUSDT', 'DASHUSDT', 'HUSDT', 'ARBUSDT', 'CRVUSDT', 'CVCUSDT', 'VIRTUALUSDT', '币安人生USDT', 'HIFIUSDT', 'ICPUSDT', 'HBARUSDT', 'STRKUSDT', 'OPUSDT', 'ZENUSDT', 'XLMUSDT', 'SNTUSDT', 'MKRUSDT', 'AIAUSDT', '1000BONKUSDT', 'LSKUSDT', 'PAXGUSDT', 'SPELLUSDT', 'BATUSDT', 'BANANAS31USDT', 'GRASSUSDT', 'FETUSDT', 'WLFIUSDT', 'SLERFUSDT', 'SEIUSDT', 'ETCUSDT', 'RAREUSDT', 'BLZUSDT', 'TONUSDT', 'MERLUSDT', 'BAKEUSDT', 'SKYAIUSDT', 'INJUSDT', 'RONINUSDT', 'STABLEUSDT', 'LDOUSDT', 'COMBOUSDT', 'DYDXUSDT', 'ONDOUSDT', 'EIGENUSDT', 'ATUSDT', 'FLUXUSDT', 'ORCAUSDT', 'CHZUSDT', 'NULSUSDT', 'LOKAUSDT', 'RESOLVUSDT', 'PARTIUSDT', 'ALLOUSDT', 'ALCHUSDT', 'METUSDT', 'ATOMUSDT', 'SPXUSDT', 'MMTUSDT', 'GALAUSDT', 'IRYSUSDT', 'KEYUSDT', 'XMRUSDT', 'ETHFIUSDT', '1INCHUSDT', 'APEUSDT', 'RECALLUSDT', 'PLUMEUSDT', 'KDAUSDT', 'PNUTUSDT', 'ARCUSDT', 'LOOMUSDT', '1000CHEEMSUSDT', 'ZKUSDT', 'OMUSDT', 'KITEUSDT', 'NEIROUSDT', 'SUPERUSDT', 'POLUSDT', 'DENTUSDT', 'TRBUSDT', 'MDTUSDT', '0GUSDT', 'SAHARAUSDT', '4USDT', 'PTBUSDT', '2ZUSDT', 'ORDIUSDT', 'PENDLEUSDT', 'SYRUPUSDT', 'NMRUSDT', 'SNXUSDT', 'TRUTHUSDT', 'LINEAUSDT', 'MELANIAUSDT', 'ALGOUSDT', 'KLAYUSDT', 'KAITOUSDT', 'OMGUSDT', 'NXPCUSDT', 'COMPUSDT', 'CAKEUSDT', 'ACTUSDT', 'USELESSUSDT', 'FOLKSUSDT', '1000FLOKIUSDT', 'HYPERUSDT', 'JUPUSDT', 'BONDUSDT', 'ZEREBROUSDT', 'CKBUSDT', 'REDUSDT', 'RENDERUSDT', 'ZROUSDT', 'HMSTRUSDT', 'REEFUSDT', 'IPUSDT', 'XEMUSDT', 'ALICEUSDT', 'SOONUSDT']
# 或者是你那完整的200个币的列表
# SYMBOLS = [...] 

START_DATE = '2024-01-01'
END_DATE = '2024-03-31'
BASE_DIR = './data/futures/um/daily/klines'
# ===========================================

# 1. 配置坚固的网络连接 (自动重试)
def get_robust_session():
    session = requests.Session()
    # 遇到错误自动重试3次，每次间隔1秒
    retry = Retry(total=3, read=3, connect=3, backoff_factor=1, 
                  status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# 全局 Session
session = get_robust_session()

def download_and_process(symbol, date_str):
    url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{date_str}.zip"
    
    save_dir = os.path.join(BASE_DIR, symbol)
    save_path = os.path.join(save_dir, f"{date_str}.parquet")
    
    # 2. 断点续传：如果文件已存在且你确定没问题，就跳过
    # 如果你想强制覆盖旧的(因为之前表头错了)，请注释掉下面这两行
    if os.path.exists(save_path):
        return "EXISTS"

    try:
        # 3. 发送请求 (连接超时5秒，读取超时20秒)
        response = session.get(url, timeout=(5, 20))
        
        if response.status_code == 404:
            return "NOT_FOUND"
        if response.status_code != 200:
            return "ERROR"

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                # 4. 读取 CSV
                df = pd.read_csv(f, header=None, low_memory=False)

                # 5. 智能表头检测
                first_cell = str(df.iloc[0, 0])
                # 如果第一格包含字母，说明是表头，删掉第一行
                if "open" in first_cell.lower() or not first_cell.replace('.', '', 1).isdigit():
                    df = df.iloc[1:]

                # 6. 删除第7列 (收盘时间)
                # 原始是12列(0-11)，我们要删掉索引为6的那列
                if df.shape[1] >= 7:
                    df = df.drop(columns=[6,11])  # 删除第7列和第12列

                # 7. 重命名剩余的 11 列
                df.columns = [
                    'timestamp',            # 0. 开盘时间
                    'open',                 # 1. 开盘价
                    'high',                 # 2. 最高价
                    'low',                  # 3. 最低价
                    'close',                # 4. 收盘价
                    'volume',               # 5. 成交量
                    'quote_volume',         # 7. 成交额 (重要)
                    'count',                # 8. 笔数 (重要)
                    'taker_buy_volume',     # 9. 主动买入量
                    'taker_buy_quote_volume' # 10. 主动买入额 (重要)
                ]
                
                # 8. 强制转为数字
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df.dropna(inplace=True)
                if df.empty: return "EMPTY"

                # 9. 时间戳转换
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                os.makedirs(save_dir, exist_ok=True)
                df.to_parquet(save_path, index=False)
                return "SUCCESS"
                
    except Exception as e:
        print(f"\nError on {symbol} {date_str}: {str(e)}")
        return "ERROR"

def main():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    date_range = [start + timedelta(days=x) for x in range((end - start).days + 1)]

    total_tasks = len(SYMBOLS) * len(date_range)
    print(f"计划下载 {len(SYMBOLS)} 个币种，共 {total_tasks} 个文件。")
    print("正在启动...")

    pbar = tqdm(total=total_tasks, desc="Total Progress")

    for symbol in SYMBOLS:
        for date_obj in date_range:
            date_str = date_obj.strftime("%Y-%m-%d")
            status = download_and_process(symbol, date_str)
            
            pbar.update(1)
            
            # 10. 只有成功下载才休息 (防封)
            if status == "SUCCESS":
                time.sleep(random.uniform(0.1, 0.3)) 
            
    pbar.close()
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()