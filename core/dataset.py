from numpy import column_stack
import torch
import torch.utils.data as data
import pandas as pd
import os
import numpy as np

class StockDataSet(data.Dataset):
    def __init__(self,data_dir, coloumns, seq_len, pred_len):
        """
        Args:
            
            coloumns (list): 用来预测的指标["open","high","low","close","pre_close","pct_chg","vol","turnover_rate","volume_ratio"]
                                        [开盘价，最高价，最低价，收盘价，昨日收盘价，价格变化百分比(%)，成交量（手），成交额（千元），换手率（%），量比]
            seq_len (int): 用过去多少天的数据来预测
            pred_len (int): 预测未来几天的收盘价
        """
        self.data_dir = data_dir
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.sample_need_len = seq_len + pred_len
        self.need_open_normlize_columns = ["open","high","low","close","pre_close"]
        self.need_self_normlize_columns = ["vol"]
        self.need_devide100_columns = ["pct_chg","turnover_rate"]
        self.columns = coloumns  
        self.column_index = {col:num for num,col in enumerate(self.columns)}
        self.samples = []
        self.norm_samples = []
        self.labels = []
        self.prepare_samples()
        self.normalize_samples()
        
    def info(self):
        data_len = len(self.norm_samples)
        print("dataset sample num is {}".format(data_len))
        label_count = {i:self.labels.count(i) for i in self.labels}
        print("label distribute \n",label_count)
        
    
    def __getitem__(self, index):
        x,y = self.norm_samples[index]
        x = torch.from_numpy(x)
        return x,y
    
    def __len__(self):
        return len(self.norm_samples)
    
    def prepare_samples(self):
        print("preparing data ... ")
        csv_files = os.listdir(self.data_dir)
        for file in csv_files:
            if "csv" in file:
                file_path = os.path.join(self.data_dir,file)
                stock_df = pd.read_csv(file_path)
                # 按日期升序排序
                stock_df = stock_df.sort_values(by="trade_date",ascending=True).reset_index(drop=True)
                rows, _ = stock_df.shape
                stock_df = stock_df.loc[:,self.columns]
                for i in range(rows - self.sample_need_len):
                    part_df = stock_df.iloc[i:i+self.sample_need_len]
                    self.samples.append(part_df)
        
        
    def normalize_samples(self):
        for sample in self.samples:
            x,y = self.normalize_sample(sample)
            # count use
            self.labels.append(y)
            self.norm_samples.append((x,y))
    
    
    def normalize_sample(self,sample):
        seq_part = sample.iloc[:self.seq_len, :]
        pred_part = sample.iloc[-self.pred_len:, :]
        arr_seq = np.array(seq_part)
        first_day_open_price = arr_seq[0, self.column_index["open"]]
        first_day_self_norm_col_values = [arr_seq[0, self.column_index[col]] for col in self.need_self_normlize_columns]
        for i in range(self.seq_len):
            for col in self.need_devide100_columns:
                arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / 100
            for col in self.need_open_normlize_columns:
                arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / first_day_open_price - 1.0
            for col in self.need_self_normlize_columns:
                 arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / first_day_self_norm_col_values[self.need_self_normlize_columns.index(col)] - 1.0
        
        y = None
        price_change = (pred_part.iloc[-1, self.column_index["close"]] - pred_part.iloc[0, self.column_index["pre_close"]]) / pred_part.iloc[0, self.column_index["pre_close"]] 
        if price_change <= -0.02:
            y = 0
        elif price_change<=0.05 and price_change >= -0.02:
            y = 1
        else:
            y = 2
        assert y is not None,"y is none"
        return arr_seq, y
        
if __name__ == "__main__":
    dataset = StockDataSet(
                            data_dir = "data/origin_data/part_qfq",
                            coloumns = ["open","high","low","close","pre_close","pct_chg","vol","turnover_rate","volume_ratio"],
                            seq_len = 5,
                            pred_len=2
                           )
    dataset.info()