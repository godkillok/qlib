import os
import sys
import pandas as pd
import akshare as ak
import baostock as bs
from datetime import datetime, timedelta
from tqdm import tqdm

ETF_DIR = "./etl_etf/etf_data.csvs"
os.makedirs(ETF_DIR, exist_ok=True)

# 获取真实交易日（使用 baostock）
def get_trade_dates(start_date, end_date):
    bs.login()
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    dates = []
    while rs.next():
        if rs.get_row_data()[1] == '1':
            dates.append(rs.get_row_data()[0])
    bs.logout()
    return dates

# 获取某天所有 ETF spot 数据
def get_etf_spot_by_date(date_str):
    try:
        df = ak.fund_etf_spot_ths(date=date_str)
        df["日期"] = date_str
        return df
    except Exception as e:
        print(f"[ERROR] 获取 {date_str} ETF 数据失败: {e}")
        return None

# 抽取目标字段，并分 ETF 保存
SAVE_COLS = ["日期", "基金代码", "基金名称", "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额"]

# 单个日期的处理逻辑
def process_one_date(date_str):
    df = get_etf_spot_by_date(date_str)
    if df is None:
        return
    df = df[SAVE_COLS]
    for _, row in df.iterrows():
        code = row["基金代码"]
        out_path = os.path.join(ETF_DIR, f"{code}.csv")
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            if date_str in existing["日期"].values:
                continue
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            existing.drop_duplicates(subset=["日期"], inplace=True)
            existing.sort_values("日期", inplace=True)
            existing.to_csv(out_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(out_path, index=False)

# 模式一：全量初始化模式（dump_all）
def run_dump_all(start_date="2010-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    trade_days = get_trade_dates(start_date, end_date)
    print(f"[dump_all] 共需处理交易日: {len(trade_days)}")
    for date_str in tqdm(trade_days):
        process_one_date(date_str)

# 模式二：每日增量补充模式（daily_insert）
def run_daily_insert():
    end_date = datetime.today().strftime("%Y-%m-%d")
    existing_dates = set()
    for fname in os.listdir(ETF_DIR):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(ETF_DIR, fname))
            existing_dates.update(df["日期"].unique())
    earliest = min(existing_dates) if existing_dates else "2023-01-01"
    trade_days = get_trade_dates(earliest, end_date)
    trade_days = [d for d in trade_days if d not in existing_dates]
    print(f"[daily_insert] 需补充交易日: {len(trade_days)}")
    for date_str in tqdm(trade_days):
        process_one_date(date_str)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qlib_etf.py [dump_all|daily_insert]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "dump_all":
        run_dump_all(start_date="2010-01-01")
    elif mode == "daily_insert":
        run_daily_insert()
    else:
        print(f"Unknown mode: {mode}, use dump_all or daily_insert")
