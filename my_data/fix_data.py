import os
import pandas as pd
from tqdm import tqdm
import shutil

# ================= 配置区域 =================
# 你的数据根目录
BASE_DIR = './data/futures/um/daily/klines'
# ===========================================

def fix_and_save(file_path):
    try:
        # 1. 读取数据
        df = pd.read_parquet(file_path)
        
        # 2. 标准化列名：将 timestamp 重命名为 open_time
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'open_time'}, inplace=True)
            
        # 3. 确保时间类型正确
        if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

        # 4. 计算补全逻辑 (强制重新计算，确保统一)
        # CloseTime = OpenTime + 1分钟 - 1毫秒
        df['close_time'] = df['open_time'] + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)
        # Ignore = 0
        df['ignore'] = 0

        # 5. 强制列顺序 (币安标准 12 列)
        target_columns = [
            'open_time',            # 1. 开盘时间
            'open',                 # 2. 开盘价
            'high',                 # 3. 最高价
            'low',                  # 4. 最低价
            'close',                # 5. 收盘价
            'volume',               # 6. 成交量
            'close_time',           # 7. 收盘时间
            'quote_volume',         # 8. 成交额
            'count',                # 9. 笔数
            'taker_buy_volume',     # 10. 主动买入量
            'taker_buy_quote_volume',# 11. 主动买入额
            'ignore'                # 12. 忽略
        ]
        
        # 检查是否所有列都存在 (防止因列名不匹配导致的报错)
        missing = [c for c in target_columns if c not in df.columns]
        if missing:
            return f"MISSING_COLS: {missing}"
            
        # 按照标准顺序重排
        df = df[target_columns]

        # 6. 安全备份逻辑
        # 原文件路径 -> 原文件路径.bak
        backup_path = file_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.move(file_path, backup_path)
        
        # 7. 保存新文件 (parquet)
        df.to_parquet(file_path, index=False)
        
        return "SUCCESS"

    except Exception as e:
        # 如果出错了，且已经生成了备份但没保存成功，尝试恢复
        # (这里简单处理，只返回错误信息)
        return f"ERROR: {str(e)}"

def main():
    print(f"🚀 开始批量修复数据: {BASE_DIR}")
    
    # 扫描所有 .parquet 文件 (排除 .bak)
    files_to_fix = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".parquet") and not file.endswith(".bak"):
                files_to_fix.append(os.path.join(root, file))
    
    print(f"📋 共扫描到 {len(files_to_fix)} 个文件。")
    
    pbar = tqdm(files_to_fix, unit="file")
    stats = {"SUCCESS": 0, "ERROR": 0, "MISSING_COLS": 0}
    
    for file_path in pbar:
        # 显示当前文件
        pbar.set_description(f"Processing {os.path.basename(file_path)}")
        
        result = fix_and_save(file_path)
        
        # 统计结果
        if result == "SUCCESS":
            stats["SUCCESS"] += 1
        elif result.startswith("MISSING"):
            stats["MISSING_COLS"] += 1
        else:
            stats["ERROR"] += 1
            print(f"\nFailed on {file_path}: {result}")

    print("\n" + "="*30)
    print("✅ 所有任务执行完毕！")
    print(f"成功修复: {stats['SUCCESS']}")
    print(f"列缺失跳过: {stats['MISSING_COLS']}")
    print(f"发生错误: {stats['ERROR']}")
    print("="*30)
    print("\n💡 提示：")
    print("1. 原文件已重命名为 .bak 保存在同目录下。")
    print("2. 请随机抽查几个文件，确保数据正确。")

if __name__ == "__main__":
    main()