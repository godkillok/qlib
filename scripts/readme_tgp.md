# 一次性补数据任务

## 补k线数据
python qlib_baostock.py dump_all

## 补code to name
python3 qlib_baostock.py   get_stock_code


# 天级数据任务

## step1. 补k线数据
python qlib_baostock.pydaily_insert

## step2  计算
python3  calculate_daily.py

## step3  人工check
python3  stock_board.py 

