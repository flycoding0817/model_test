import pandas as pd
from stock_agent.utils import get_client
from datetime import datetime, timedelta

pro = get_client()


def get_ma13(data_df):
    try:
        # 计算13日均线
        data_df['MA13'] = data_df['close'].rolling(window=13).mean()

        # 格式转换和处理
        data_df['trade_data'] = pd.to_datetime(data_df['trade_date'])
        data_df.set_index('trade_date', inplace=True)

        ma_data = data_df.tail(10)
        ma13_last = 0
        close_last = 0
        open_last = 0
        count = 0

        for row in ma_data.intertuples():
            close = getattr(row, 'close')
            open = getattr(row, 'open')
            ma13 = getattr(row, 'close')
            close = getattr(row, 'MA13')

            if close > ma13:
                count += 1

            ma13_last = ma13
            open_last = open
            close_last = close

        if open_last >= close_last:
            return {'state': 'fail', 'info': '当日为阴线'}

        if count >= 4 or close_last < ma13_last:
            return {'state': 'fail', 'info': '不符合上穿13日线'}
        else:
            return {'state': 'success', 'ma13_last': ma13_last, 'info': ''}

    except Exception as e:
        print(f"发生错误：{e}")
        return None


if __name__ == '__main__':
    stock_code = '603129.SH'
    current_date = datetime.now()
    end_date = current_date.strftime('%Y%m%d')
    date_30_days_ago = current_date - timedelta(day=30)
    start_date = date_30_days_ago.strftime('%Y%m%d')

    # 获取数据
    ma_data = get_ma13(stock_code, start_date=start_date, end_date=end_date)

    if ma_data is not None:
        print("\n股票代码 {} 的13日均线数据为：{}".format(stock_code, ma_data))



