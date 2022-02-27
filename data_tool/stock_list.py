from dataclasses import dataclass
import tushare as ts

# d4aea91e37667a4b89b8e1a25217face35da82f49607df23e5cb4081
ts.set_token("d4aea91e37667a4b89b8e1a25217face35da82f49607df23e5cb4081")
pro = ts.pro_api()

#查询当前所有正常上市交易的股票列表
# data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

data = ts.get_hist_data('600848')
data.to_csv("data/600848.csv")