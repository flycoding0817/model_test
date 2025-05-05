from stock_agent.utils import get_client
from datetime import datetime, timedelta


def call_macd_metric(data,short_=9,long_=21,m=6):
    '''
    data是包含高开低收成交量的标准dataframe
    short_,long_,m分别是macd的三个参数
    返回值是包含原始数据和diff,dea,macd三个列的dataframe
    '''
    data == data.sort_values('trade_date')
    data['diff']=data['close'].ewm(adjust=False,alpha=2/(short_+1),ignore_na=True).mean()-\
                data['close'].ewm(adjust=False,alpha=2/(long_+1),ignore_na=True).mean()
    data['dea']=data['diff'].ewm(adjust=False,alpha=2/(m+1),ignore_na=True).mean()
    data['macd']=2*(data['diff']-data['dea'])
    macd = data['macd'].dropna().tolist()
    diff = data['diff'].dropna().tolist()
    dea = data['dea'].dropna().tolist()
    return {'macd': macd, 'diff': diff, 'dea': dea}


if __name__ == '__main__':
    pro = get_client()
    stock_code = '603129.SH'
    current_date = datetime.now()
    end_date = current_date.strftime('%Y%m%d')
    date_30_days_ago = current_date - timedelta(day=30)
    start_date = date_30_days_ago.strftime('%Y%m%d')

    # 获取数据
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    macd_dict = call_macd_metric(df, 9, 21, 6)
    print(f"macd_dict:{macd_dict}")
