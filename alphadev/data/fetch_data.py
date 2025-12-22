import pandas as pd
import os
import gzip
from pathlib import Path
from io import BytesIO

def read_parquet_gz(file_path, columns=None):
    """
    通用 Parquet 读取函数 (增强版)。
    支持 .parquet (ZSTD/Snappy) 和 .parquet.gz (Snappy+GZIP) 格式。
    """
    file_path_str = str(file_path)
    
    # 路径容错查找
    if not os.path.exists(file_path_str):
        if file_path_str.endswith('.gz'):
            alt_path = file_path_str[:-3]
        else:
            alt_path = file_path_str + '.gz'
            
        if os.path.exists(alt_path):
            file_path_str = alt_path
        # 如果都不存在，这里不抛错，让后续 pandas 或调用方处理，或者抛出更明确的错
        elif not os.path.exists(file_path_str):
             raise FileNotFoundError(f"找不到文件 (尝试了 .parquet 和 .parquet.gz): {file_path}")

    try:
        # .parquet.gz: 双重压缩 (Snappy Parquet inside GZIP)
        if file_path_str.endswith('.parquet.gz'):
            with gzip.open(file_path_str, 'rb') as f_in:
                parquet_data = BytesIO(f_in.read())
                df = pd.read_parquet(parquet_data, columns=columns, engine='pyarrow')
            return df

        # .parquet: 标准 Parquet (通常是 ZSTD 或 Snappy)
        df = pd.read_parquet(file_path_str, columns=columns, engine='pyarrow')
        return df
    except Exception as e:
        print(f"读取失败: {file_path_str}")
        if "Repetition level" in str(e):
            print("提示: 这是一个底层格式错误，通常意味着文件下载不完整或磁盘损坏。")
        raise e

def save_df_to_parquet(df: pd.DataFrame, file_path: Path | str):
    """
    通用 Parquet 保存函数。
    根据文件后缀自动选择压缩方式：
    - .parquet.gz -> 内存中写入 Snappy Parquet，然后 GZIP 压缩写入磁盘 (兼容旧格式)。
    - .parquet    -> 直接使用 ZSTD 压缩写入磁盘 (新格式，速度快)。
    """
    path_str = str(file_path)
    path_obj = Path(file_path)
    
    # 确保父目录存在
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if path_str.endswith('.parquet.gz'):
        # 模拟旧逻辑：先转成 parquet bytes (snappy)，再 gzip
        buf = BytesIO()
        df.to_parquet(buf, engine='pyarrow', compression='snappy')
        buf.seek(0)
        with gzip.open(path_obj, 'wb') as f_out:
            f_out.write(buf.getvalue())
    else:
        # 默认使用 ZSTD，如果文件名没后缀，强制加 .parquet
        if not path_str.endswith('.parquet'):
            path_obj = path_obj.with_suffix('.parquet')
        df.to_parquet(path_obj, engine='pyarrow', compression='zstd')

__all__ = ["read_parquet_gz", "save_df_to_parquet"]