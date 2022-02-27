from datetime import date
import pandas as pd

# 读取数据
df = pd.read_csv("data/600848.csv")

# columns ['date', 'open', 'high', 'close', 'low', 'volume', 'price_change','p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20','turnover']
# 每一列的名称
columns = df.columns
# index 
# 每一行的名称
index = df.index
print(index)

# shape
rows, cols = df.shape
print("rows = {}, cols = {}".format(rows,cols))


date_column = df["date"]
print("date_column[0] = " ,date_column[0])
print("last element of date is ",date_column[rows-1])

# add a new column average
df["average"] = (df["high"] + df["low"])/2

df["flag"] = df["close"] > df["open"]

# delete column "average"
del df["average"]

# use assign to create new column, not inplace
df = df.assign(shock=(df["high"] - df["low"])/df["open"])

# select condition
print(df[df["shock"]>0.1])


part_df = df[['open', 'high', 'close', 'low']]
print(part_df)

# select a row -> series
print("part_df.loc[0]\n", part_df.loc[0])
print("part_df.iloc[0]\n", part_df.iloc[0])

# first_row.index == df.columns
first_row = df.iloc[0]
print("first_row.loc['open'] = ",first_row.loc['open'])
print("df.iloc[0,1] = ",df.iloc[0,1])
# select rows -> dateframe
print("df[2:10] = ", df[2:10])

print(df.loc[2:5,['open', 'high', 'close', 'low']])

# By integer slices, acting similar to NumPy/Python:
print(df.iloc[2:5,1:3])

print(df.iloc[[1,3,5],[1,6,9,11]])


# return array
high_values = df["high"].values

print("max(df['high']) = ", max(df["high"]))
