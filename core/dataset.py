from numpy import column_stack
import torch
import torch.utils.data as data
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt

class StockDataSet(data.Dataset):
    def __init__(self,name, data_dir, coloumns, seq_len, pred_len):
        """
        Args:
            
            coloumns (list): 用来预测的指标["open","high","low","close","pre_close","pct_chg","vol","turnover_rate","volume_ratio"]
                                        [开盘价，最高价，最低价，收盘价，昨日收盘价，价格变化百分比(%)，成交量（手），成交额（千元），换手率（%），量比]
            seq_len (int): 用过去多少天的数据来预测
            pred_len (int): 预测未来几天的收盘价
        """
        self.name = name
        self.data_dir = data_dir
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.sample_need_len = seq_len + pred_len
        self.need_open_normlize_columns = ["open","high","low","close","pre_close"]
        self.need_self_normlize_columns = ["vol"]
        # self.need_devide100_columns = ["pct_chg","turnover_rate"]
        self.columns = coloumns  
        self.column_index = {col:num for num,col in enumerate(self.columns)}
        # self.norm_samples = []
        self.norm_samples = multiprocessing.Manager().list()
        self.prepare_samples()  # 将数据拆分成样本,并归一化
        self.info()
        
    def info(self):
        data_len = len(self.norm_samples)
        print("{} dataset sample num is {}".format(self.name, data_len))
        all_labels = [sample[-1] for sample in self.norm_samples]
        set_labels = set(all_labels)
        label_count = {i:all_labels.count(i) for i in set_labels}
        print(f"{self.name} dataset label distribute \n",label_count)
        
    
    def __getitem__(self, index):
        x,y = self.norm_samples[index]
        x = torch.from_numpy(x).type(torch.float32)
        return x,y
    
    def __len__(self):
        return len(self.norm_samples)
    
    def prepare_samples(self):
        print(f"preparing {self.name} data ... ")
        csv_files = os.listdir(self.data_dir)
        samples = []
        for file in tqdm(csv_files):
            if "csv" in file:
                file_path = os.path.join(self.data_dir,file)
                stock_df = pd.read_csv(file_path)
                # 按日期升序排序
                stock_df = stock_df.sort_values(by="trade_date",ascending=True).reset_index(drop=True)
                rows, _ = stock_df.shape
                stock_df = stock_df.loc[:,self.columns]
                for i in range(rows - self.sample_need_len):
                    sample = stock_df.iloc[i:i+self.sample_need_len]
                    samples.append(sample)
        self.multiprocess_normalize_samples(samples)            
        
        
    def multiprocess_normalize_samples(self,samples):
        with Pool(40) as p:
            p.map(self.normalize_sample,samples)

        
    def normalize_sample(self,sample):
        
        seq_part = sample.iloc[:self.seq_len, :]
        pred_part = sample.iloc[-self.pred_len:, :]
        arr_seq = np.array(seq_part)
        first_day_open_price = arr_seq[0, self.column_index["open"]]
        first_day_self_norm_col_values = [arr_seq[0, self.column_index[col]] for col in self.need_self_normlize_columns if col in self.columns]
        for i in range(self.seq_len):
            # for col in self.need_devide100_columns:
            #     if col in self.columns:
            #         arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / 100
            for col in self.need_open_normlize_columns:
                if col in self.columns:
                    arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / first_day_open_price - 1.0
            for col in self.need_self_normlize_columns:
                if col in self.columns:
                    arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / first_day_self_norm_col_values[self.need_self_normlize_columns.index(col)] - 1.0
        
        y = 0
        price_change = (pred_part.iloc[-1, self.column_index["close"]] - pred_part.iloc[0, self.column_index["pre_close"]]) / pred_part.iloc[0, self.column_index["pre_close"]] 
        if price_change <= 0.0:
            y = 0
        else:
            y = 1
        # if price_change < -0.08 or price_change > 0.08:
        #     plt.figure()
        #     plt.plot(arr_seq)
        #     plt.legend(self.columns)
        #     plt.title(str(price_change))
        #     plt.savefig("log/visualization/{}/{}.png".format(y,str(price_change)))
        assert y is not None,"y is none"
        self.norm_samples.append((arr_seq, y))
        
if __name__ == "__main__":
    dataset = StockDataSet(
                            name="train",
                            data_dir = "data/test_qfq",
                            coloumns = ["open","high","low","close","pre_close","pct_chg","vol","turnover_rate","volume_ratio"],
                            seq_len = 5,
                            pred_len = 2
                           )
    dataset.info()