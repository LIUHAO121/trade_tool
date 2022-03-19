import os
import pandas as pd
import time
from tqdm import tqdm
import tushare as ts



ts.set_token("ac33207e59b0c3702ee846ade596f11c88b793cd3cb28431467f9ec8")
pro = ts.pro_api()


def stock_basic():
    save_path = "data/stock_basic.csv"
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    return df



def get_qfq(ts_code,start_date,end_date):
    time.sleep(1)
    df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date,factors=['tor', 'vr'])
    return df

def get_all_company_qfq(end_date):
    stock_basic_df = stock_basic()
    rows, cols = stock_basic_df.shape
    for row in tqdm(range(rows)):
        ts_code  = stock_basic_df.loc[row,"ts_code"]
        list_date = stock_basic_df.loc[row,"list_date"]
        save_path = "data/origin_data/qfq/{}_{}_{}_qfq.csv".format(ts_code,list_date,end_date)
        # if not os.path.exists(save_path):
        company_qfq = get_qfq(ts_code, str(list_date), end_date)
        if company_qfq is not None:
            company_qfq_dropna = company_qfq.dropna(axis=0,how="any")
            company_qfq_dropna.to_csv(save_path)
            print("saving {} ".format(save_path))
        
        

if __name__ == "__main__":
    ts_code = '000001.SZ'
    start_date = '20000101'    
    end_date = '20220305'
    get_all_company_qfq(end_date)
    
    