import pandas as pd
import os
from pathlib import Path

def read_parquet_gz(file_path, columns=None):
    """
    通用 Parquet 读取函数 (增强版)。
    1. 兼容 pathlib.Path 和 str 路径。
    2. 支持按列读取 (极大提升速度，减少读取坏块的概率)。
    3. 自动处理后缀查找。
    """
    # 1. 统一转为 string 处理路径
    file_path_str = str(file_path)
    
    # 2. 路径存在性检查与容错
    if not os.path.exists(file_path_str):
        # 尝试去掉或添加 .gz 后缀寻找替代文件
        if file_path_str.endswith('.gz'):
            alt_path = file_path_str[:-3] # 去掉 .gz
        else:
            alt_path = file_path_str + '.gz' # 加上 .gz
            
        if os.path.exists(alt_path):
            file_path_str = alt_path
        else:
            raise FileNotFoundError(f"找不到文件: {file_path_str}")

    # 3. 读取数据 (关键修改：传入 columns)
    try:
        # 即使 columns 里有不需要的列，pandas 也会忽略多余的，
        # 但如果 columns 为 None，则读取所有。
        df = pd.read_parquet(
            file_path_str, 
            columns=columns, 
            engine='pyarrow'
        )
        return df
    except Exception as e:
        # 这里不要只打印，打印完抛出异常，让外层捕获
        print(f"读取 Parquet 文件失败: {file_path_str}")
        # 有时候 pyarrow 报错很含糊，加一句提示
        if "Repetition level" in str(e):
            print("提示: 这是一个底层格式错误，通常意味着文件下载不完整或磁盘损坏。")
        raise e