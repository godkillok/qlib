import dash
from dash import dcc, html, Input, Output, State, callback
import dash_ag_grid as dag
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
import qlib
from qlib.data import D
from qlib.constant import REG_CN
import numpy as np
from datetime import datetime

# 初始化QLib数据源
provider_uri = "/Users/tanggp/qlib_data/"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 读取CSV文件并合并生成股票列表
df1 = pd.read_csv('/Users/tanggp/qlib_raw/stock_code.csv')  # 包含 code,tradeStatus,code_name,symbol
df2 = pd.read_csv(
    "/Users/tanggp/Documents/quanta/qlib/scripts/angle_stocks.csv")  # 包含 code,last_close,close_angle,ma5_angle,volume,is_not_st,status_ok

# 合并数据并筛选结果
merged = pd.merge(
    df2[["code", "close_angle", "ma5_angle"]],
    df1[["symbol", "code_name"]],
    left_on="code",
    right_on="symbol",
    how="inner"
)
stocks = merged[["code", "close_angle", "ma5_angle", "code_name"]].to_dict(orient='records')


# 定义获取股票数据的函数
def get_stock_data(instrument, start_date, end_date):
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    kline_data = D.features([instrument], fields, start_time=start_date, end_time=end_date)
    df = kline_data.loc[instrument].reset_index()
    df = df.tail(30)
    df["date"] = pd.to_datetime(df["datetime"])
    df["ma5"] = talib.SMA(df["$close"], timeperiod=5)
    df["ma10"] = talib.SMA(df["$close"], timeperiod=10)
    df["ma20"] = talib.SMA(df["$close"], timeperiod=20)
    upper, middle, lower = talib.BBANDS(df["$close"], timeperiod=20)
    df["Upper"] = upper
    df["Middle"] = middle
    df["Lower"] = lower
    df["RSI"] = talib.RSI(df["$close"], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df["$close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macdsignal
    df["MACD_Hist"] = macdhist
    slowk, slowd = talib.STOCH(df["$high"], df["$low"], df["$close"])
    df["K"] = slowk
    df["D"] = slowd
    df["J"] = 3 * df["K"] - 2 * df["D"]
    df["pct_change"] = df["$close"].pct_change() * 100
    df["is_up"] = df["$close"] > df["$open"]
    return df


# 创建Dash应用
app = dash.Dash(__name__)

# 定义表格列
grid_columns = [
    {"field": "code_name", "headerName": "股票名称", "sortable": True, "filter": True},
    {"field": "code", "headerName": "代码", "sortable": True, "filter": True},
    {"field": "close_angle", "headerName": "收盘角度", "sortable": True, "filter": True},
    {"field": "ma5_angle", "headerName": "MA5角度", "sortable": True, "filter": True},
]

app.layout = html.Div([
    html.Div([
        html.H2("股票分析系统", style={'textAlign': 'center'}),

        # 搜索区域
        html.Div([
            dcc.Input(
                id="stock-search",
                type="text",
                placeholder="输入股票名称或代码搜索...",
                style={"width": "100%", "padding": "8px", "marginBottom": "10px"}
            ),
        ]),

        # 虚拟滚动表格
        dag.AgGrid(
            id="stock-grid",
            # 基础参数
            columnDefs=grid_columns,
            rowData=stocks,
            defaultColDef={"sortable": True, "filter": True, "resizable": True},

            # 网格配置
            dashGridOptions={
                "pagination": True,
                "paginationPageSize": 20,
                "rowSelection": "single",  # 单行选择模式
                "suppressRowClickSelection": True,  # 可选：需要点击复选框才能选中
                "cacheBlockSize": 100,
                "maxBlocksInCache": 2
            },

            # 样式参数
            # rowHeight=35,
            className="ag-theme-alpine",
            style={"height": "600px", "width": "100%"},

            # 选择状态
            selectedRows=[],  # 必须的参数用于接收选中数据
        ),

        html.Div(id="selected-stock-info", style={"marginTop": "20px"})
    ], style={"width": "30%", "padding": "20px", "borderRight": "1px solid #ddd"}),

    html.Div([
        # 信息显示容器
        html.Div(
            id='hover-info',
            style={
                'position': 'absolute',
                'top': '20px',
                'left': '50%',
                'transform': 'translateX(-50%)',
                'background': 'rgba(255, 255, 255, 0.95)',
                'padding': '10px 15px',
                'border-radius': '8px',
                'box-shadow': '0 2px 10px rgba(0,0,0,0.15)',
                'border': '1px solid #eee',
                'z-index': '100',
                'display': 'none',
                'font-family': 'Arial, sans-serif',
                'font-size': '14px',
                'min-width': '280px'
            }
        ),

        # K线图
        dcc.Graph(
            id='main-chart',
            style={'height': '50vh', 'position': 'relative'},
            config={'displayModeBar': False}
        ),

        # 子图1
        dcc.Graph(
            id='sub-chart1',
            style={'height': '25vh'}
        ),

        # 子图2
        dcc.Graph(
            id='sub-chart2',
            style={'height': '25vh'}
        )
    ], style={'width': '70%', 'padding': '20px', 'position': 'relative'}),

    # 隐藏存储
    dcc.Store(id='stock-data'),
    dcc.Store(id='selected-stock', data=None)
], style={'display': 'flex', 'height': '100vh'})


# 回调函数：搜索股票
@app.callback(
    Output("stock-grid", "rowData"),
    Input("stock-search", "value"),
    State("stock-grid", "rowData")
)
def search_stocks(search_term, all_stocks):
    if not search_term:
        return stocks
    search_term = search_term.lower()
    filtered_stocks = [
        stock for stock in stocks
        if search_term in stock["code_name"].lower() or search_term in stock["code"]
    ]
    return filtered_stocks


# 回调函数：显示选中的股票信息
@app.callback(
    [Output("selected-stock-info", "children"),
     Output("selected-stock", "data")],
    Input("stock-grid", "selectedRows")
)
def display_selected_stock(selected_rows):
    if selected_rows:
        selected_stock = selected_rows[0]
        return [
            html.Div([
                html.H4(f"选中的股票：{selected_stock['code_name']} ({selected_stock['code']})"),
                html.P(f"收盘角度：{selected_stock['close_angle']}", style={"margin": "5px 0"}),
                html.P(f"MA5角度：{selected_stock['ma5_angle']}", style={"margin": "5px 0"})
            ]),
            selected_stock["code"]
        ]
    return [html.Div("未选择股票"), None]


# 回调函数：加载股票数据
@app.callback(
    Output('stock-data', 'data'),
    Input('selected-stock', 'data')
)
def load_stock_data(stock_code):
    if not stock_code:
        return None
    # 根据选中的股票代码获取数据
    instrument = stock_code  # 假设股票代码可以直接作为instrument
    df = get_stock_data(instrument, "2025-01-01", "2025-05-30")
    return df.to_dict('records')


# 回调函数：更新价格信息
@app.callback(
    [Output('current-price', 'children'),
     Output('price-change', 'children')],
    Input('stock-data', 'data')
)
def update_price_info(data):
    if not data:
        return "", ""
    df = pd.DataFrame(data)
    last_row = df.iloc[-1]
    price = f"{last_row['$close']:.2f}"
    pct_change = last_row['pct_change']
    change_text = f"{'↑' if pct_change >= 0 else '↓'} {abs(pct_change):.2f}%"
    color = 'red' if pct_change >= 0 else 'green'
    return price, html.Span(change_text, style={'color': color})


# 回调函数：更新主图
@app.callback(
    Output('main-chart', 'figure'),
    [Input('stock-data', 'data'),
     Input('period-selector', 'value'),
     Input('main-indicator-selector', 'value')]
)
def update_main_chart(data, period, indicators):
    if not data:
        return go.Figure()

    try:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])

        # 根据周期转换数据
        if period == 'weekly':
            agg_dict = {
                '$open': 'first', '$high': 'max', '$low': 'min', '$close': 'last',
                '$volume': 'sum', 'ma5': 'last', 'ma10': 'last', 'ma20': 'last',
                'Upper': 'last', 'Middle': 'last', 'Lower': 'last'
            }
            df = df.resample('W-FRI', on='date').agg(agg_dict).reset_index()
        elif period == 'monthly':
            agg_dict = {
                '$open': 'first', '$high': 'max', '$low': 'min', '$close': 'last',
                '$volume': 'sum', 'ma5': 'last', 'ma10': 'last', 'ma20': 'last',
                'Upper': 'last', 'Middle': 'last', 'Lower': 'last'
            }
            df = df.resample('M', on='date').agg(agg_dict).reset_index()

        fig = go.Figure()
        latest_30_dates = df['date'].iloc[-10:]
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['$open'],
                high=df['$high'],
                low=df['$low'],
                close=df['$close'],
                name='K线',
                # range=[  # 初始显示范围设置
                #     latest_30_dates.iloc[0].strftime('%Y-%m-%d'),
                #     latest_30_dates.iloc[-1].strftime('%Y-%m-%d')
                # ],
            )
        )

        if 'ma' in indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['ma5'], name='MA5', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['ma10'], name='MA10', line=dict(color='green', width=1)))

        if 'boll' in indicators and 'Upper' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['Upper'], name='上轨', line=dict(color='blue', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['Lower'], name='下轨', line=dict(color='blue', width=1, dash='dot'), fill='tonexty'))

        # 智能抽样函数（确保至少显示首尾日期）
        def smart_sample(dates, target=10):
            if len(dates) <= target:
                return dates, dates.dt.strftime('%Y%m%d')

            selected = [0, -1]
            # 均匀抽样中间日期
            step = max(1, (len(dates) - 2) // (target - 2))
            selected += list(range(1, len(dates) - 1, step))

            sampled_dates = dates.iloc[selected]
            return sampled_dates, sampled_dates.dt.strftime('%Y%m%d')

        # 获取抽样后的日期和标签
        tick_dates, tick_labels = smart_sample(df['date'])

        fig.update_layout(
            xaxis=dict(
                type='category',
                tickmode='array',
                tickvals=tick_dates,
                ticktext=tick_labels,
                tickangle=-45,
                rangeslider_visible=True,
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True,
            hovermode='x unified'

        )

        return fig
    except Exception as e:
        print(f"Error updating main chart: {str(e)}")
        return go.Figure()

    # 回调函数：更新子图1


# 回调函数：更新子图1
@app.callback(
    Output('sub-chart1', 'figure'),
    [Input('stock-data', 'data'),
     Input('period-selector', 'value'),
     Input('subchart1-selector', 'value')]
)
def update_subchart1(data, period, indicator):
    if not data:
        return go.Figure()
    # 这里的逻辑保持不变，直接沿用您原有的实现
    # ...


# 回拨函数：更新子图2
@app.callback(
    Output('sub-chart2', 'figure'),
    [Input('stock-data', 'data'),
     Input('period-selector', 'value'),
     Input('subchart2-selector', 'value')]
)
def update_subchart2(data, period, indicator):
    # 重用update_subchart1的逻辑
    return update_subchart1(data, period, indicator)


# 回调函数：显示悬停信息
@app.callback(
    Output('hover-info', 'children'),
    Output('hover-info', 'style'),
    Input('main-chart', 'hoverData'),
    State('stock-data', 'data')
)
def display_hover_data(hoverData, stock_data):
    if not hoverData or not stock_data:
        return None, {'display': 'none'}
    # 这里的逻辑保持不变，直接沿用您原有的实现

    try:
        point_index = hoverData['points'][0]['pointIndex']
        df = pd.DataFrame(stock_data)
        row = df.iloc[point_index]
        last_row = df.iloc[-1]  # 获取最后一行数据

        # 确保日期是datetime对象
        if not isinstance(row['date'], pd.Timestamp):
            row['date'] = pd.to_datetime(row['date'])

        # 计算各类涨跌幅
        pct_change = (row['$close'] - row['$open']) / row['$open'] * 100
        final_pct_change = (last_row['$close'] - row['$close']) / row['$close'] * 100
        vol_wan = row['$volume'] / 10000
        amount_yi = row['$volume'] * row['$close'] / 100000000

        def format_volume(vol):
            if vol >= 100000000:
                return f"{vol / 100000000:.2f}亿"
            elif vol >= 10000:
                return f"{vol / 10000:.0f}万"
            return f"{vol:.0f}"

        # 调整后的布局（放大字体并添加最终涨幅）
        info_card = html.Div([
            # 第一行：日期 + 价格 + 当日涨跌幅
            html.Div([
                html.Span(
                    row['date'].strftime('%Y/%m/%d') + " ",
                    style={'margin-right': '10px', 'font-size': '20px'}
                ),
                html.Span(
                    f"{row['$close']:.2f} ",
                    style={
                        'font-weight': 'bold',
                        'font-size': '22px',  # 价格再放大
                        'margin-right': '8px'
                    }
                ),
                html.Span(
                    f"{'↑' if pct_change >= 0 else '↓'}{abs(pct_change):.2f}%",
                    style={
                        'color': '#e00' if pct_change >= 0 else '#0a0',
                        'font-weight': 'bold',
                        'font-size': '20px'
                    }
                )
            ], style={'margin-bottom': '8px'}),

            # 第二行：关键指标（放大字体）
            html.Div([
                html.Div([
                    html.Span(
                        f"开:{row['$open']:.2f} ",
                        style={'font-size': '18px', 'margin-right': '12px'}
                    ),
                    html.Span(
                        f"高:{row['$high']:.2f} ",
                        style={'font-size': '18px', 'margin-right': '12px'}
                    ),
                    html.Span(
                        f"低:{row['$low']:.2f}",
                        style={'font-size': '18px'}
                    )
                ]),
                html.Div([
                    html.Span(
                        f"量:{format_volume(row['$volume'])} ",
                        style={'font-size': '18px', 'margin-right': '12px'}
                    ),
                    html.Span(
                        f"额:{amount_yi:.2f}亿",
                        style={'font-size': '18px'}
                    )
                ])
            ], style={'margin-bottom': '8px', 'line-height': '1.5'}),

            # 新增第三行：距最后时刻涨幅（大号字体）
            html.Div([
                html.Span(
                    "距最新价: ",
                    style={'font-size': '18px', 'margin-right': '6px'}
                ),
                html.Span(
                    f"{'↑' if final_pct_change >= 0 else '↓'}{abs(final_pct_change):.2f}%",
                    style={
                        'color': '#e00' if final_pct_change >= 0 else '#0a0',
                        'font-size': '18px',
                        'font-weight': 'bold',
                        'margin-right': '6px'
                    }
                ),
                html.Span(
                    f"(最新:{last_row['$close']:.2f})",
                    style={'font-size': '16px', 'color': '#666'}
                )
            ])
        ], style={
            'min-width': '350px',  # 增加宽度容纳放大字体
            'padding': '15px',  # 增加内边距
            'background': 'rgba(255, 255, 255, 0.98)'  # 更不透明
        })

        return info_card, {
            'display': 'block',
            'position': 'absolute',
            'top': '15px',
            'left': '50%',
            'transform': 'translateX(-50%)',
            'background': 'rgba(255, 255, 255, 0.98)',
            'border-radius': '10px',  # 更大圆角
            'box-shadow': '0 3px 12px rgba(0,0,0,0.2)',  # 更明显阴影
            'z-index': '100',
            'border': '1px solid #eee',
            'font-family': 'Arial, sans-serif'
        }

    except Exception as e:
        print(f"Hover error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, {'display': 'none'}


if __name__ == '__main__':
    app.run(debug=True, port=8050)