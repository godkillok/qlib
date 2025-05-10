# -*- coding: utf-8 -*-
import os
import logging
import baostock as bs
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import sys
import  time
from tqdm import tqdm  # 导入tqdm
print(Path(__file__).resolve().parent.joinpath("scripts"))
'''
/Users/tanggp/Documents/work/qlib-main/scripts/dump_bin.py
/Users/tanggp/Documents/work/qlib-main/qlib/contrib/scripts
'''

sys.path.append("/Users/tanggp/Documents/work/qlib-main/scripts")
from dump_bin import DumpDataAll, DumpDataUpdate


class Config:
    # Baostock 参数
    BAOSTOCK_USER = "anonymous"  # 默认公共账号
    BAOSTOCK_PASSWORD = "123456"  # 默认密码
    ADJUST_TYPE = "2"  # 复权类型: 1-后复权 2-前复权 3-不复权

    # Qlib 参数
    QLIB_DIR = "~/qlib_data"  # Qlib数据存储目录
    RAW_DATA_DIR = "~/qlib_raw"  # 原始CSV数据目录
    FREQ = "d"  # 数据频率: d/1m/5m
    FREQ_QLIAB="day"

    # 市场类型 (all/csi300/csi500)"all"#
    MARKET = "all"#"csi300"

    # 网络请求参数
    MAX_RETRY = 3  # 单只股票下载重试次数
    TIMEOUT = 10  # 请求超时时间(秒)


class QlibBaostockIntegration:
    def __init__(self):
        self.cfg = Config()
        self._init_paths()
        self._login_baostock()

    def _init_paths(self) -> None:
        """初始化目录结构"""
        self.qlib_dir = Path(self.cfg.QLIB_DIR).expanduser()
        self.raw_data_dir = Path(self.cfg.RAW_DATA_DIR).expanduser()
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def _login_baostock(self) -> None:
        """登录Baostock（支持重试）"""
        for _ in range(self.cfg.MAX_RETRY):
            try:
                self.bs = bs.login(
                    # user_id=self.cfg.BAOSTOCK_USER,
                    # password=self.cfg.BAOSTOCK_PASSWORD,
                    # timeout=self.cfg.TIMEOUT
                )
                if self.bs.error_code == '0':
                    return
                time.sleep(2)
            except Exception as e:
                logging.warning(f"登录失败: {str(e)}, 重试中...")
        raise ConnectionError("无法连接Baostock服务器")

    def _convert_code_format(self, code: str) -> str:
        """代码格式转换: Baostock(sh.600000) <-> Qlib(sh600000)"""
        if "." in code:  # Baostock -> Qlib
            return code.replace(".", "")
        else:  # Qlib -> Baostock
            return f"{code[:2]}.{code[2:]}"

    def _get_last_trading_date(self) -> str:
        """获取最后一个有效交易日（通过query_trade_dates）"""
        # 查询最近30天的日期（避免周末/节假日导致无数据）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != '0':
            raise ValueError(f"获取交易日历失败: {rs.error_msg}")

        # 筛选交易日并排序
        trading_dates = sorted(
            [row[0] for row in rs.data if row[1] == '1'],  # is_trading_day=1
            reverse=True
        )

        if not trading_dates:
            raise ValueError("未找到有效交易日")
        return trading_dates[0]  # 返回最近的交易日

    def _get_symbols(self) -> List[str]:
        """获取目标股票列表（动态更新成分股）"""
        date = self._get_last_trading_date()
        # extra=[ "sh000300"
        # ]
        #
        # extra=[self._convert_code_format(i) for i in extra]
        extra=[]
        if self.cfg.MARKET == "csi300":
            rs = bs.query_hs300_stocks(date=date)
            return extra+[self._convert_code_format(row[1]) for row in rs.data]
        elif self.cfg.MARKET == "csi500":
            rs = bs.query_zz500_stocks(date=date)
            return extra+[self._convert_code_format(row[1]) for row in rs.data]
        else:
            rs = bs.query_all_stock(day=date)
            return extra+[self._convert_code_format(row[0]) for row in rs.data]



    def _download_single_stock(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """下载单只股票数据（自动重试）"""
        bs_code = self._convert_code_format(symbol)
        # print(bs_code,start,end,self.cfg.FREQ,self.cfg.ADJUST_TYPE)
        for _ in range(self.cfg.MAX_RETRY):
            try:
                rs = bs.query_history_k_data_plus(
                    code=bs_code,
                    #"date", "open", "close", "high", "low", "volume", "money", "change"
                    fields="date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                    start_date=start,
                    end_date=end,
                    frequency=self.cfg.FREQ,
                    adjustflag=self.cfg.ADJUST_TYPE
                )
                # print(rs.data)
                if rs.error_code != '0':
                    continue

                df = pd.DataFrame(rs.data, columns=rs.fields)
                df["symbol"] = symbol
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date")
            except Exception as e:
                print(f"下载 {symbol} 失败: {str(e)}")
                time.sleep(1)
        return None

    def _get_trading_dates(self, start: str, end: str) -> List[str]:
        """获取有效交易日历（避免下载非交易日数据）"""
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        dates = [row[0] for row in rs.data if row[1] == '1']
        return dates

    def dump_all(self, years: int = 20) -> None:
        """全量模式：下载历史数据并导入Qlib"""
        symbols = self._get_symbols()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

        # 下载全量数据
        print("股票数量",len(symbols))
        for symbol in tqdm(symbols, desc="股票下载进度", unit="只"):
            csv_path = self.raw_data_dir / f"{symbol}.csv"
            if csv_path.exists():
                continue

            df = self._download_single_stock(symbol, start_date, end_date)
            if df is not None and not df.empty:
                df.to_csv(csv_path, index=False)
                logging.info(f"已保存: {csv_path}")

        # 导入Qlib
        DumpDataAll(
            csv_path=str(self.raw_data_dir),
            qlib_dir=str(self.qlib_dir),
            freq=self.cfg.FREQ_QLIAB,
            date_field_name="date",
            instrucment_file_name=self.cfg.MARKET,
            symbol_field_name="symbol",
            exclude_fields="code,symbol",
        ).dump()

    def daily_insert(self) -> None:
        """增量模式：更新当日数据"""
        # 获取Qlib最新日期
        calendar_file = self.qlib_dir / "calendars" / f"{self.cfg.FREQ_QLIAB}.txt"

        if not calendar_file.exists():
            raise FileNotFoundError(calendar_file,"请先运行dump_all初始化数据")

        with open(calendar_file, "r") as f:
            last_date = max(line.strip() for line in f)

        start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        trading_dates = self._get_trading_dates(start_date, end_date)

        if not trading_dates:
            logging.info("无新交易日数据")
            return

        下载增量数据
        symbols = self._get_symbols()
        for symbol in tqdm(symbols, desc="股票下载进度", unit="只"):
            csv_path = self.raw_data_dir / f"{symbol}.csv"
            df_old = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

            df_new = self._download_single_stock(symbol, start_date, end_date)
            if df_new is None or df_new.empty:
                continue

            df_combined = pd.concat([df_old, df_new]).drop_duplicates("date")
            df_combined["date"]=pd.to_datetime(df_combined["date"],format='mixed')
            df_combined.to_csv(csv_path, index=False)

        # 增量导入Qlib
        # 导入Qlib
        DumpDataAll(
            csv_path=str(self.raw_data_dir),
            qlib_dir=str(self.qlib_dir),
            freq=self.cfg.FREQ_QLIAB,
            date_field_name="date",
            instrucment_file_name=self.cfg.MARKET,
            symbol_field_name="symbol",
            exclude_fields="code,symbol",
        ).dump()
        # DumpDataUpdate(
        #     csv_path=str(self.raw_data_dir),
        #     qlib_dir=str(self.qlib_dir),
        #     freq=self.cfg.FREQ_QLIAB,
        #     date_field_name="date",
        #     symbol_field_name="symbol",
        #     instrucment_file_name=self.cfg.MARKET,
        #     exclude_fields="code,symbol",
        # ).dump()

        # DumpDataAll(
        #     csv_path=str(self.raw_data_dir),
        #     qlib_dir=str(self.qlib_dir),
        #     freq=self.cfg.FREQ_QLIAB,
        #     date_field_name="date",
        #     instrucment_file_name=self.cfg.MARKET,
        #
        #     symbol_field_name="symbol",
        #     exclude_fields="code,symbol",
        # ).dump()

    def __del__(self):
        """确保登出"""
        if hasattr(self, 'bs'):
            bs.logout()


if __name__ == "__main__":
    import fire
    '''
    python3  qlib_baostock.py  dump_all
    python3  qlib_baostock.py  daily_insert
    '''
    fire.Fire(QlibBaostockIntegration)




