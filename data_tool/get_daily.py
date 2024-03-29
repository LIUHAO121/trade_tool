from ast import expr_context
import tushare as ts
import time
from tqdm import tqdm


ts.set_token("ac33207e59b0c3702ee846ade596f11c88b793cd3cb28431467f9ec8")
pro = ts.pro_api()


def get_daily(date,is_open):
    if is_open:
        for i in range(3):
            try:
                df = pro.daily(trade_date=date)
                return df
            except:
                print("try {} time failed for {} data".format(i+1, date))
                time.sleep(5)
    else:
        return None
    
    
# 交易日历
def get_trade_cal(start,end):
    df = pro.trade_cal(exchange='', start_date=start, end_date=end)
    return df


def get_multiple_daily(start_date,end_date):
    # 先获取交易日历,确定哪天是交易日
    df_cal = get_trade_cal(start_date,end_date)
    df_cal.to_csv("data/basic/trade_cal_{}_{}.csv".format(start_date,end_date))

    rows = len(df_cal)
    for i in tqdm(range(rows)):
        row = df_cal.iloc[i]
        date = row["cal_date"]
        is_open = row["is_open"]
        trade_daily = get_daily(date,is_open)
        if trade_daily is not None:
            trade_daily.to_csv("data/origin_data/date_organized/{}.csv".format(date))
            print("successfully get {} data".format(date))
            
def get_one_company_data(ts_code,start_date,end_date):
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df.to_csv("data/{}_{}_{}.csv".format(ts_code,start_date,end_date))

if __name__ == "__main__":
    start_date = "20000101"
    end_date = "20010220"
    # get_multiple_daily(start_date=start_date,end_date=end_date)
    get_one_company_data("000001.SZ",start_date=start_date,end_date=end_date)