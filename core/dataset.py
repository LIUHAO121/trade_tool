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

import math

# def log_encode(y):
#     y = max(y,-0.99)
#     return math.log(y + 1)

# def exp_decode(predict):
#     return math.exp(predict) - 1

def log_encode(y):
    return y

def exp_decode(predict):
    return predict
    

class StockClsDataSet(data.Dataset):
    def __init__(self,dataset_type, data_dir, coloumns, seq_len, pred_len,split=0.8):
        """
        Args:
            
            coloumns (list): 用来预测的指标["open","high","low","close","pre_close","pct_chg","vol","turnover_rate","volume_ratio"]
                                        [开盘价，最高价，最低价，收盘价，昨日收盘价，价格变化百分比(%)，成交量（手），成交额（千元），换手率（%），量比]
            seq_len (int): 用过去多少天的数据来预测
            pred_len (int): 预测未来几天的收盘价
        """
        self.split = split
        self.dataset_type = dataset_type
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
        print("{} dataset sample num is {}".format(self.dataset_type, data_len))
        all_labels = [sample[-1] for sample in self.norm_samples]
        set_labels = set(all_labels)
        label_count = {i:all_labels.count(i) for i in set_labels}
        print(f"{self.dataset_type} dataset label distribute \n",label_count)
        
    
    def __getitem__(self, index):
        x,y = self.norm_samples[index]
        x = torch.from_numpy(x).type(torch.float32)
        return x,y
    
    def __len__(self):
        return len(self.norm_samples)
    
    def prepare_samples(self):
        print(f"preparing {self.dataset_type} data ... ")
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
                train_rows = int(rows * self.split)
                if self.dataset_type == "train":
                    stock_df = stock_df.iloc[:train_rows]
                    rows = train_rows
                elif self.dataset_type == "test":
                    stock_df = stock_df.iloc[train_rows:]
                    rows = rows - train_rows
                else:
                    raise ValueError(f"unrecognize dataset type: {self.dataset_type}") 
                for i in range(0, rows - self.sample_need_len, 1):
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
        # if price_change < -0.05 or price_change > 0.05:
            
        #     plt.figure()
        #     plt.plot(arr_seq)
        #     plt.legend(self.columns)
        #     plt.title(str(price_change))
        #     plt.savefig("log/visualization/{}/{}.png".format(y,str(price_change)))
        self.norm_samples.append((arr_seq, y))
    
class StockRegDataSet(data.Dataset):
    def __init__(self,dataset_type, data_dir, coloumns, seq_len, pred_len,split=0.8):
        """
        Args:
            coloumns (list): 用来预测的指标["open","high","low","close","pre_close","pct_chg","vol","turnover_rate","volume_ratio"]
                                        [开盘价，最高价，最低价，收盘价，昨日收盘价，价格变化百分比(%)，成交量（手），成交额（千元），换手率（%），量比]
            seq_len (int): 用过去多少天的数据来预测
            pred_len (int): 预测未来几天的收盘价
        """
        self.split = split
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.sample_need_len = seq_len + pred_len
        self.need_open_normlize_columns = ["open","high","low","close","pre_close"]
        self.need_self_normlize_columns = ["vol"]
        self.columns = coloumns  
        self.column_index = {col:num for num,col in enumerate(self.columns)}
        self.norm_samples = multiprocessing.Manager().list()
        self.prepare_samples()  # 将数据拆分成样本,并归一化
        
    def info(self):
        data_len = len(self.norm_samples)
        print("{} dataset sample num is {}".format(self.dataset_type, data_len))

    
    def __getitem__(self, index):
        x,y = self.norm_samples[index]
        x = torch.from_numpy(x).type(torch.float32)
        # 当y为一个单独的元素时，把y放到列表里，保证shape为（batch，1）或（batch，n），而不是（batch，）,交叉墒的y的shape可以是这样
        y = torch.tensor([y]).type(torch.float32) 
        return x,y
    
    def __len__(self):
        return len(self.norm_samples)
    
    def prepare_samples(self):
        print(f"preparing {self.dataset_type} data ... ")
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
                train_rows = int(rows * self.split)
                if self.dataset_type == "train":
                    stock_df = stock_df.iloc[:train_rows]
                    rows = train_rows
                elif self.dataset_type == "test":
                    
                    stock_df = stock_df.iloc[train_rows:]
                    rows = rows - train_rows
                else:
                    raise ValueError(f"unrecognize dataset type: {self.dataset_type}") 
                self.stock_df = stock_df
                for i in range(0, rows - self.sample_need_len, 1):
                    sample = stock_df.iloc[i:i+self.sample_need_len]
                    samples.append(sample)
        self.multiprocess_normalize_samples(samples)            
        
        
    def multiprocess_normalize_samples(self,samples):
        with Pool(40) as p:
            p.map(self.normalize_sample,samples)

        
    def normalize_sample(self,sample):
        # 以样本的长度为周期计算y
        arr_seq = np.array(sample)
        first_day_open_price = arr_seq[0, self.column_index["close"]]
        first_day_self_norm_col_values = [arr_seq[0, self.column_index[col]] for col in self.need_self_normlize_columns if col in self.columns]
        for i in range(self.sample_need_len):
            for col in self.need_open_normlize_columns:
                if col in self.columns:
                    arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / first_day_open_price - 1.0
            for col in self.need_self_normlize_columns:
                if col in self.columns:
                    arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / first_day_self_norm_col_values[self.need_self_normlize_columns.index(col)] - 1.0
        
        x = arr_seq[:self.seq_len]
        y = arr_seq[-1,self.column_index["close"]]
        
        y = log_encode(y)

        self.norm_samples.append((x, y))
        

    def norm_sample_fun(self,sample):
        rows,_ = sample.shape
        arr_seq = np.array(sample)
        first_day_open_price = arr_seq[0, self.column_index["close"]]
        first_day_self_norm_col_values = [arr_seq[0, self.column_index[col]] for col in self.need_self_normlize_columns if col in self.columns]
        for i in range(rows):
            for col in self.need_open_normlize_columns:
                if col in self.columns:
                    arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / first_day_open_price - 1.0
            for col in self.need_self_normlize_columns:
                if col in self.columns:
                    arr_seq[i,self.column_index[col]] = arr_seq[i,self.column_index[col]] / first_day_self_norm_col_values[self.need_self_normlize_columns.index(col)] - 1.0
        x = arr_seq[:self.seq_len]
        y = arr_seq[-1,self.column_index["close"]]
        y = log_encode(y)
        return x,y

    
    
    def predict_sequences_multiple_dense(self,model,interval):
        print("predict sequences multiple dense ... ")
        model.eval()
        model.cuda()
        rows,_ = self.stock_df.shape
        pred_nums = int((rows - self.sample_need_len)/interval) + 1
        prediction_seqs = []
        gt_values = []
        real_values = list(self.stock_df.iloc[:,self.column_index["close"]].values)
        with torch.no_grad():
            for i in range(pred_nums - 1):
                predicted_unnorm_values = []
                start_point = i * interval
                part_stock_df = self.stock_df.iloc[start_point:start_point + self.sample_need_len, :]
                input_norm_sample,y = self.norm_sample_fun(part_stock_df)     
                for j in range(self.seq_len):                 
                    input_tensor = torch.from_numpy(input_norm_sample[np.newaxis,:,:]).type(torch.float32).cuda()
                    
                    out = model(input_tensor).cpu()[0][0].item()
                    out = exp_decode(out)
                    
                    input_norm_sample = input_norm_sample[1:]
                    input_norm_sample = np.insert(input_norm_sample, self.seq_len - 1, out, axis=0) 
                    predicted_unnorm_values.append(out)
                prediction_seqs.append(predicted_unnorm_values)
    
        return  real_values,prediction_seqs
    
    def predict_point_by_point(self,model):
        print("predict sequences point by point ... ")
        model.eval()
        model.cuda()
        rows,_ = self.stock_df.shape
        predicts = []
        ground_truth_values = []
        with torch.no_grad():
            for i in range(rows - self.sample_need_len):
                predicted_unnorm_values = []
                part_stock_df = self.stock_df.iloc[i :i  + self.sample_need_len ,:]
                input_norm_sample, y = self.norm_sample_fun(part_stock_df)
                input_tensor = torch.from_numpy(input_norm_sample[np.newaxis,:,:]).type(torch.float32).cuda()
                out = model(input_tensor).cpu()[0][0].item()
                out = exp_decode(out)
                predicts.append(out)
                ground_truth_values.append(y)
        return  ground_truth_values, predicts
    
    def test_predict(self,model):
        print("predict test predict ... ")
        model.eval()
        model.cuda()
        part_df = self.stock_df.iloc[-self.seq_len:,:]
        input_norm_sample,y = self.norm_sample_fun(part_df)
        predict_out = []
        for i in range(self.seq_len):
            input_tensor = torch.from_numpy(input_norm_sample[np.newaxis,:,:]).type(torch.float32).cuda()
            out = model(input_tensor).cpu()[0][0].item()
            out = exp_decode(out)
            input_norm_sample = input_norm_sample[1:]
            insert_index = self.seq_len - 1
            input_norm_sample = np.insert(input_norm_sample, insert_index, out, axis=0) 
            predict_out.append(out)
        past_real_values = list(self.stock_df.iloc[-self.seq_len*4:,self.column_index["close"]].values)
        return past_real_values, predict_out
    
    def test_predict_dense(self,model,interval,num_interval):
        """
        interval: 间隔多久预测一次
        num_seq_len:  预测多少次
        """
        print("predict test predict ... ")
        model.eval()
        model.cuda()
        need_history_df_len = self.seq_len + interval * (num_interval - 1)
        need_history_df = self.stock_df.iloc[-need_history_df_len:,:]
        predict_outs = []
        for j in range(num_interval):
            part_df = need_history_df.iloc[j*interval:j*interval+self.seq_len,:]
            input_norm_sample,y = self.norm_sample_fun(part_df)
            predict_out = []
            for i in range(self.seq_len):
                input_tensor = torch.from_numpy(input_norm_sample[np.newaxis,:,:]).type(torch.float32).cuda()
                out = model(input_tensor).cpu()[0][0].item()
                out = exp_decode(out)
                input_norm_sample = input_norm_sample[1:]
                insert_index = self.seq_len - 1
                input_norm_sample = np.insert(input_norm_sample, insert_index, out, axis=0) 
                predict_out.append(out)
            predict_outs.append(predict_out)
        past_real_values = list(need_history_df.iloc[:,self.column_index["close"]].values)
        return past_real_values, predict_outs
            
if __name__ == "__main__":
    dataset = StockClsDataSet(
                            dataset_type="train",
                            data_dir = "data/test_qfq",
                            coloumns = ["open","high","low","close","pre_close","pct_chg","vol","turnover_rate","volume_ratio"],
                            seq_len = 5,
                            pred_len = 2
                           )
    dataset.info()