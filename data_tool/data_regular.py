import pandas as pd
import os

def pct_chg(csv_path):
    """当一只股票某天的振幅超过10%，第二天上涨的概率

    Args:
        csv_path (str): 某只股票的历史数据
        df_columns: ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
    """
    df = pd.read_csv(csv_path)
    rows,cols = df.shape
    
    up_count = 0
    pct_chg_10_count = 0
    for row in range(rows-1,0,-1):
        open,high,low,close = df.loc[row,["open","high","low","close"]]
        change = high - low
        change_pct = change/open
        next_day_close = df.loc[row - 1,"close"]
        if abs(change_pct) > 0.1 and close > open:
           pct_chg_10_count += 1
           if next_day_close > close:
               up_count += 1
            #    print("pct_chg_10 {} up {}".format(pct_chg_10_count,up_count))
    return up_count/(pct_chg_10_count + 1e-5)


def m5(df,row):
    """
    计算某天的五日均线，即当前天与前四天的收盘价之和
    """
    rows, cols = df.shape
    sum_ = 0
    if (rows - 4) >= row:
        for i in range(5):
            sum_ += df.loc[row+i,"close"]
        return sum_ / 5
    else:
        return df.loc[row,"close"]
            
        

def m5_strategy(csv_path,f_n):
    """股票超过五日均线后上涨的概率

    Args:
        csv_path (str): 某只股票的历史数据
        df_columns: ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
        f_n : 观察未来n天的变化
    """
    df = pd.read_csv(csv_path)
    rows,cols = df.shape
    
    up_count = 0
    satisfy_count = 0
    
    for row in range(rows-6,f_n,-1):
        open,high,low,close = df.loc[row,["open","high","low","close"]]

        today_m5 = m5(df,row)
        last_day_m5 = m5(df,row + 1) # 昨日m5
        last_day_close = df.loc[row + 1, "close"]
        
        if close > today_m5 and last_day_close < last_day_m5:
            satisfy_count += 1
            for j in range(1,f_n):
                next_n_day_close = df.loc[row-j,"close"]
                if (next_n_day_close - close)/close > 0.05:
                    up_count += 1
                    break
    return up_count , satisfy_count
        

if __name__ == "__main__":
    csv_dirs = "data/origin_data/qfq"
    csv_files = os.listdir(csv_dirs)
    for file in csv_files:
        csv_path = os.path.join(csv_dirs,file)
        up_count , satisfy_count = m5_strategy(csv_path=csv_path,f_n=10)
        pct = up_count/(satisfy_count+0.0001)
        print("{}:  {} / {} = {}".format(file[:9],up_count,satisfy_count,pct))
        