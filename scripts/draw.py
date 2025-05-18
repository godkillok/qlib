import qlib
from qlib.data import D
from qlib.constant import REG_CN
import plotly.graph_objects as go
import pandas as pd
import talib
from talib import RSI, MACD, STOCH
import dash
from dash import dcc, html, Input, Output, callback, State
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# 初始化QLib数据源
provider_uri = "/Users/tanggp/qlib_data/"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 股票列表
# stocks = [
#     {'code': 'sh601012', 'name': '隆基绿能',"info1":"info1","info2":"info2"},
#     {'code': 'sh600519', 'name': '贵州茅台',"info1":"info1","info2":"info2"},
#     {'code': 'sz300750', 'name': '宁德时代',"info1":"info1","info2":"info2"},
#     {'code': 'sh688041', 'name': '贵州',"info1":"info1","info2":"info2"},
#     {'code': 'sz000158', 'name': '宁',"info1":"info1","info2":"info2"}
# ]

import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('/Users/tanggp/qlib_raw/stock_code.csv')  # 包含 code,tradeStatus,code_name,symbol
df2 = pd.read_csv("/Users/tanggp/Documents/quanta/qlib/scripts/angle_stocks.csv")  # 包含 code,last_close,close_angle,ma5_angle,volume,is_not_st,status_ok

# 使用右连接 (保留csv2中所有code)
merged = pd.merge(
    df2[["code", "close_angle", "ma5_angle"]],  # 只选择csv2需要的列
    df1[["symbol", "code_name"]],              # 只选择csv1需要的列
    left_on="code",                            # 用csv2.code匹配
    right_on="symbol",                         # csv1.symbol
    how="inner"                                # 以csv2的code为基准
)

# 重命名列并筛选结果
result = merged[["code", "close_angle", "ma5_angle", "code_name"]][:20]
print(result)
stocks = result.to_dict(orient='records')

# 获取股票数据并计算指标
def get_stock_data(instrument, start_date, end_date):
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    kline_data = D.features([instrument], fields, start_time=start_date, end_time=end_date)
    df = kline_data.loc[instrument].reset_index()
    df =df.tail(30)
    # 过滤掉没有交易数据的日期（如周末、节假日）
    df = df[df["$close"].notna()]  # 确保收盘价不为空
    df["date"] = pd.to_datetime(df["datetime"])

    # 计算均线
    df["ma5"] = talib.SMA(df["$close"], timeperiod=5)
    df["ma10"] = talib.SMA(df["$close"], timeperiod=10)
    df["ma20"] = talib.SMA(df["$close"], timeperiod=20)

    # 计算布林带
    upper, middle, lower = talib.BBANDS(df['$close'], timeperiod=20)
    df['Upper'] = upper
    df['Middle'] = middle
    df['Lower'] = lower

    # RSI
    df['RSI'] = RSI(df['$close'], timeperiod=14)

    # MACD
    macd, macdsignal, macdhist = MACD(df['$close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    # KDJ
    slowk, slowd = STOCH(df['$high'], df['$low'], df['$close'])
    df['K'] = slowk
    df['D'] = slowd
    df['J'] = 3 * df['K'] - 2 * df['D']

    df['pct_change'] = df['$close'].pct_change() * 100
    df['is_up'] = df['$close'] > df['$open']  # 添加涨跌判断

    return df

# 创建Dash应用
app = dash.Dash(__name__, external_stylesheets=[
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
])

app.layout = html.Div([
    html.Div([
        # 左侧控制面板
        html.Div([
            html.H2("股票分析系统", style={'textAlign': 'center'}),

            # 股票选择列表 - 改为上下布局
            html.Div([
                html.H4("选择股票"),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                html.Div([
                                    html.Div([  # 名称与代码行
                                        html.Span(s['code_name'], style={'fontWeight': 'bold'}),
                                        html.Span(f"({s['code'][2:]})",
                                                  style={'fontSize': '0.8em', 'marginLeft': '6px'}),
                                        #close_angle	ma5_angle
                                        html.Span(s['close_angle'], style={'flex': 1, 'marginLeft': '6px'}),
                                        html.Span(s['ma5_angle'], style={'flex': 1, 'marginLeft': '6px'}),
                                    ])
                                ], className='stock-item',
                                    id={'type': 'stock-item', 'index': s['code']},
                                    n_clicks=0),
                                style={
                                    'padding': '8px',
                                    'cursor': 'pointer',
                                    'borderBottom': '1px solid #eee',
                                    'display': 'block'  # 保持原始display模式
                                }
                            ) for s in stocks
                        ])
                    ])
                ], style={'width': '100%'})
            ], style={'margin': '20px 0'}),

            # 时间周期选择
            html.Div([
                html.H4("时间周期"),
                dcc.RadioItems(
                    id='period-selector',
                    options=[
                        {'label': '日K', 'value': 'daily'},
                        {'label': '周K', 'value': 'weekly'},
                        {'label': '月K', 'value': 'monthly'}
                    ],
                    value='daily',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                )
            ], style={'margin': '20px 0'}),

            # 主图指标选择
            html.Div([
                html.H4("主图指标"),
                dcc.Checklist(
                    id='main-indicator-selector',
                    options=[
                        {'label': '均线', 'value': 'ma'},
                        {'label': '布林线', 'value': 'boll'}
                    ],
                    value=['ma'],
                    labelStyle={'display': 'block'}
                )
            ], style={'margin': '20px 0'}),

            # 子图1指标选择
            html.Div([
                html.H4("子图1指标"),
                dcc.Dropdown(
                    id='subchart1-selector',
                    options=[
                        {'label': 'RSI', 'value': 'RSI'},
                        {'label': 'MACD', 'value': 'MACD'},
                        {'label': 'KDJ', 'value': 'KDJ'},
                        {'label': '成交量', 'value': 'VOL'}
                    ],
                    value='RSI',
                    clearable=False
                )
            ], style={'margin': '20px 0'}),

            # 子图2指标选择
            html.Div([
                html.H4("子图2指标"),
                dcc.Dropdown(
                    id='subchart2-selector',
                    options=[
                        {'label': 'RSI', 'value': 'RSI'},
                        {'label': 'MACD', 'value': 'MACD'},
                        {'label': 'KDJ', 'value': 'KDJ'},
                        {'label': '成交量', 'value': 'VOL'}
                    ],
                    value='MACD',
                    clearable=False
                )
            ], style={'margin': '20px 0'}),

            # 当前价格信息
            html.Div([
                html.H4("当前价格"),
                html.Div(id='current-price', style={'fontSize': 24}),
                html.Div(id='price-change', style={'fontSize': 16})
            ])
        ], style={'width': '20%', 'padding': '20px', 'borderRight': '1px solid #ddd'}),

        # 右侧图表区
        html.Div([
            # 信息显示容器（放在图表容器内部）
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
        ], style={'width': '80%', 'padding': '20px', 'position': 'relative'})
    ], style={'display': 'flex', 'height': '100vh'}),

    # 隐藏存储
    dcc.Store(id='stock-data'),
    dcc.Store(id='selected-stock', data=stocks[0]['code'])
])


# 回调函数：处理股票选择
@callback(
    [Output('selected-stock', 'data'),
     Output({'type': 'stock-item', 'index': dash.ALL}, 'style')],
    [Input({'type': 'stock-item', 'index': dash.ALL}, 'n_clicks')],
    [State({'type': 'stock-item', 'index': dash.ALL}, 'id')]
)
def select_stock(clicks, items):
    ctx = dash.callback_context
    if not ctx.triggered:
        return stocks[0]['code'], [{}] * len(stocks)

    stock_code = ctx.triggered[0]['prop_id'].split('"index":"')[1].split('"')[0]

    styles = []
    for item in items:
        if item['index'] == stock_code:
            styles.append({'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'})
        else:
            styles.append({})

    return stock_code, styles


# 回调函数：加载股票数据
@callback(
    Output('stock-data', 'data'),
    Input('selected-stock', 'data')
)
def load_stock_data(stock_code):
    df = get_stock_data(stock_code, "2025-01-01", "2025-05-30")
    return df.to_dict('records')


# 回调函数：更新价格信息
@callback(
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
@callback(
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


@callback(
    Output('sub-chart1', 'figure'),
    [Input('stock-data', 'data'),
     Input('period-selector', 'value'),
     Input('subchart1-selector', 'value')]
)
def update_subchart1(data, period, indicator):
    if not data:
        return go.Figure()

    try:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])

        if period == 'weekly':
            agg_dict = {
                '$volume': 'sum', 'RSI': 'last', 'MACD': 'last', 'MACD_Signal': 'last',
                'MACD_Hist': 'last', 'K': 'last', 'D': 'last', 'J': 'last', 'is_up': 'last'
            }
            df = df.resample('W-FRI', on='date').agg(agg_dict).reset_index()
        elif period == 'monthly':
            agg_dict = {
                '$volume': 'sum', 'RSI': 'last', 'MACD': 'last', 'MACD_Signal': 'last',
                'MACD_Hist': 'last', 'K': 'last', 'D': 'last', 'J': 'last', 'is_up': 'last'
            }
            df = df.resample('M', on='date').agg(agg_dict).reset_index()

        fig = go.Figure()

        if indicator == 'RSI':
            fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', line=dict(color='purple')))
            fig.update_yaxes(range=[0, 100])

        elif indicator == 'MACD':
            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['MACD_Hist'],
                name='MACD Hist',
                marker_color=np.where(df['MACD_Hist'] < 0, 'red', 'green')
            ))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_Signal'], name='Signal', line=dict(color='orange')))

        elif indicator == 'KDJ':
            fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['J'], name='J', line=dict(color='purple')))
            fig.update_yaxes(range=[0, 100])

        elif indicator == 'VOL':
            colors = ['red' if up else 'green' for up in df['is_up']]
            fig.add_trace(go.Bar(x=df['date'], y=df['$volume'], name='成交量', marker_color=colors))
            fig.update_layout(xaxis_type='category')

        fig.update_layout(
            xaxis=dict(
                showticklabels=False,
                showline=True,
                linecolor='lightgray',
                mirror=True,
                ticks=''
            ),
            margin=dict(l=20, r=20, t=20, b=0),
            showlegend=True
        )

        return fig
    except Exception as e:
        print(f"Error updating subchart1: {str(e)}")
        return go.Figure()

# 回调函数：更新子图2
@callback(
    Output('sub-chart2', 'figure'),
    [Input('stock-data', 'data'),
     Input('period-selector', 'value'),
     Input('subchart2-selector', 'value')]
)
def update_subchart2(data, period, indicator):
    # 重用update_subchart1的逻辑
    return update_subchart1(data, period, indicator)


@app.callback(
    Output('hover-info', 'children'),
    Output('hover-info', 'style'),
    Input('main-chart', 'hoverData'),
    State('stock-data', 'data')
)
def display_hover_data(hoverData, stock_data):
    if not hoverData or not stock_data:
        return None, {'display': 'none'}

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