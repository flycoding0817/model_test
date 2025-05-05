from stock_agent.utils import get_client
from datetime import datetime, timedelta


def call_boll_bands(data, window=26, n=2):
    data = data.sort_values('trade_date')
    data["mid"] = data["close"].rolling(window=window).mean
    data["upper"] = data["mid"] + data["close"].rolling(window=window).std() * n
    data["lower"] = data["mid"] - data["close"].rolling(window=window).std() * n

    mid = data['mid'].dropna().tolist()
    upper = data['upper'].dropna().tolist()
    lower = data['lower'].dropna().tolist()

    return {'mid': mid, 'upper': upper, 'lower': lower}


if __name__ == '__main__':
    pro = get_client()
    stock_code = '603129.SH'
    current_date = datetime.now()
    end_date = current_date.strftime('%Y%m%d')
    date_30_days_ago = current_date - timedelta(day=30)
    start_date = date_30_days_ago.strftime('%Y%m%d')

    # 获取数据
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    macd_dict = call_boll_bands(df, 20, 2)
    print(f"macd_dict:{macd_dict}")
