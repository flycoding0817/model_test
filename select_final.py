import time
from datetime import datetime, timedelta
from stock_agent.metric_compute.macd import call_macd_metric
from stock_agent.metric_compute.boll import call_boll_bands
from stock_agent.metric_compute.avg13 import get_ma13
from stock_agent.utils import get_client
from stock_agent.utils import check_macd_positive
from stock_agent.utils import is_increaseing_trend_up
pro = get_client()


stock_dict = {'000997.SH': '新大陆'}

current_date = datetime.now()
end_date = current_date.strftime('%Y%m%d')
date_30_days_ago = current_date - timedelta(day=30)
start_date = date_30_days_ago.strftime('%Y%m%d')

count = 0

for ts_code, ts_name in stock_dict.items():
    time.sleep(1)
    count += 1

    # 获取最近60天数据
    data_df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    data_df = data_df.sort_values('trade_date')

    # 最近交易日和收盘价
    last_date_row = data_df.iloc[-1]
    last_date = last_date_row['trade_date']
    last_price = last_date_row['close']
    print("{}: {}({})最新交易日期：{}, 收盘价：{}".format(count, ts_name, ts_code, last_date, last_price))

    # 13日均价判断
    ma13_data_dict = get_ma13(data_df)
    if ma13_data_dict['state'] == 'success':
        ma13_data = ma13_data_dict['ma13_last']
    else:
        info = ma13_data_dict['info']
        print('{} 代码：{} {}'.format(stock_dict[ts_code], ts_code, info))

    # macd 判断
    macd_dict = call_macd_metric(data_df)
    macd = macd_dict['macd']
    if not check_macd_positive(macd):
        print('{} 代码：{} macd 量能未符合要求'.format(stock_dict[ts_code], ts_code))
        continue

    # boll 判断
    boll_dict = call_boll_bands(data_df)
    boll_upper = boll_dict['upper']
    boll_mid = boll_dict['mid']
    boll_lower = boll_dict['lower']

    # if not (is_increaseing_trend_up(boll_upper) and is_increaseing_trend_up(boll_lower)):
    #     print('代码：{} boll 边界未能趋势向上'.format(ts_code))
    #     continue

    if last_price < boll_mid[-1]:
        print('{} 代码：{} boll mid 未能站上均线'.format(stock_dict[ts_code], ts_code))
        continue

    print('----------{} 代码：{} 符合目标，请注意查看！')
