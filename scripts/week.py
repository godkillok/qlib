import os
import pandas as pd
import talib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# 配置参数
DATA_FOLDER = Path("/Users/tanggp/qlib_raw")  # CSV文件存放路径
RESULT_FILE = './angle_stocks.csv'  # 结果输出文件
TRADE_DAYS = 21  # 需要分析的历史交易日数
KLINE_NUM = 5  # 计算最近K线数量


def calculate_relative_angle(stock_code,prices, time_unit=1):
    """
    基于相对涨幅的角度计算
    :param prices: 原始价格序列
    :param time_unit: 时间单位（日=1，周=5）
    :return: 角度（度）
    """
    if len(prices) < 2:
        return np.nan
    if "301209" in stock_code:
        print(stock_code)
    # 转换为相对涨幅（百分比）
    base_price = prices[-1]
    relative_prices = (prices / base_price) * 100  # 基准化为100

    # 计算线性回归斜率
    slope = talib.LINEARREG_SLOPE(relative_prices, timeperiod=len(relative_prices))[-1]

    # 角度计算（时间单位宽度固定为1）
    angle = np.degrees(np.arctan(slope / 1))  # 因为Y轴已经是百分比单位
    return angle


def calculate_angle(prices, time_unit=1):
    """
    计算价格序列的上升角度
    :param prices: 价格序列（需至少5个数据点）
    :param time_unit: 每个K线的时间单位（日K线=1）
    :return: 角度（度）
    """
    if len(prices) < KLINE_NUM or np.isnan(prices).any():
        return np.nan

    try:
        # 计算线性回归斜率
        slope = talib.LINEARREG_SLOPE(prices[-KLINE_NUM:], timeperiod=KLINE_NUM)[-1]

        # 计算价格比例因子（使用最近价格的1%作为基准）
        price_scale = prices[-1] * 0.01

        # 转换为角度（反正切计算）
        angle = np.degrees(np.arctan(slope / (price_scale * time_unit)))
        return angle
    except Exception as e:
        print(f"角度计算失败: {str(e)}")
        return np.nan


def process_stock_file(file_path):
    """处理单个股票文件"""
    try:
        # 从文件名提取股票代码
        stock_code = os.path.basename(file_path).split('.')[0]

        # 读取数据
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date')#.tail(TRADE_DAYS)  # 取最近数据

        if len(df) < KLINE_NUM:
            return None

        # 数据预处理
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('float64')
        df = df.dropna(subset=['close', 'high', 'low', 'volume'])

        # 计算技术指标
        closes = df['close'].values
        df['MA5'] = talib.MA(closes, timeperiod=5)

        # 计算最近5根K线的角度（收盘价和MA5分别计算）
        df=df.tail(KLINE_NUM)
        close_angle = calculate_relative_angle(stock_code,df['close'].values)

        ma5_angle = calculate_relative_angle(stock_code,df['MA5'].values)


        last_high = df['high'].iloc[-1]
        prev_high = df['high'].iloc[-2]
        last_close = df['close'].iloc[-1]
        last_open = df['open'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        prev_open = df['open'].iloc[-2]
        df['downtrend'] = 0
        if last_high < prev_high and last_close < last_open and prev_close < prev_open:
            df['downtrend'] = 1

        # 获取最新数据
        latest_data = {
            'code': stock_code,
            'last_close': closes[-1],
            'close_angle': close_angle,
            'ma5_angle': ma5_angle,
            'volume': df['volume'].iloc[-1],
            'is_not_st': df['isST'].iloc[-1] != 0,
            'status_ok': df['tradestatus'].iloc[-1] == 1,
            'downtrend': df['downtrend'].iloc[-1] if len(df) > 0 else 0
        }
        if  "002891" in stock_code :
            print(latest_data)
        return latest_data
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None


def batch_process():
    """批量处理所有股票文件"""
    results = []

    # 获取所有CSV文件
    csv_files = [f for f in DATA_FOLDER.iterdir() if f.suffix == '.csv']
    valid=0
    # 使用tqdm添加进度条
    for file_path in tqdm(csv_files, desc="处理股票", unit="只"):
        result = process_stock_file(file_path)
        if result:
            results.append(result)

    # 保存结果
    if results:
        result_df = pd.DataFrame(results)
        # 筛选有效数据
        result_df = result_df[
            (result_df['close_angle'] > 0) &
            (result_df['ma5_angle']*result_df['close_angle'] > 20*100) &
            (result_df['is_not_st']==0) &
            (result_df['downtrend']<1)
        ]
        # 按角度排序
        result_df = result_df.sort_values('ma5_angle', ascending=False)
        result_df.to_csv(RESULT_FILE, index=False)
        print(f"\n处理完成，找到 {len(result_df)} 只股票，结果已保存至 {RESULT_FILE}")
        print("前10名股票：")
        print(result_df[['code', 'ma5_angle', 'close_angle']].head(10))
    else:
        print("未找到符合要求的股票")


if __name__ == '__main__':
    if not DATA_FOLDER.exists():
        raise FileNotFoundError(f"数据文件夹 {DATA_FOLDER} 不存在")
    batch_process()