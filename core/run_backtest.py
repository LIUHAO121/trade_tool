import os
import argparse
import logging
import datetime as dt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from core import LSTMPred ,StockRegDataSet, BackTest,load_json
from data_tool import get_qfq



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/reg_config_close_ma5.json",
                        type=str)
    
    parser.add_argument('--ts_code',
                        help='company code',
                        default="002230.SZ",
                        type=str)
    
    parser.add_argument('--end_date',
                        help='回测最终时间',
                        default="20220430",
                        type=str)
    args = parser.parse_args()

    return args

DATASET_LIB = {
    "regression":StockRegDataSet
}


def main():
    args = parse_args()
    config = load_json(args.cfg)
    task_type = config["task_type"]
    ts_code = args.ts_code
    end_date = args.end_date
    
    log = logging.getLogger("backtest")
    log.setLevel(level = logging.INFO)
    handler = logging.FileHandler("{}/backtest_{}_{}.txt".format(config["log_dir"],ts_code, dt.datetime.now().strftime('%Y%m%d-%H%M%S')))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.info("back test {}".format(ts_code))
    
    data_dir = config["backtest"]["data_dir"]
    
    stock_basic = pd.read_csv("data/stock_basic.csv") 
    list_date = stock_basic[stock_basic["ts_code"]==ts_code]["list_date"].values[0]
    for csv_file in os.listdir(data_dir):
        csv_path = os.path.join(data_dir,csv_file)
        os.remove(csv_path)
    train_df = get_qfq(ts_code = ts_code, start_date=str(list_date), end_date=end_date)
    train_df.to_csv("{}/{}_{}.csv".format(data_dir, ts_code, end_date))
    
    dataset_task = DATASET_LIB[task_type]
    train_dataset = dataset_task(
        dataset_type = "test",
        data_dir = data_dir,
        columns = config["dataset"]["columns"],
        seq_len =  config["dataset"]["seq_len"],
        pred_len = config["dataset"]["pred_len"],
        split= config["dataset"]["train_test_split"])
    
    model = LSTMPred(
        embed_dim = config["model"]["embed_dim"],
        hidden_size = config["model"]["hidden_size"],
        rnn_layers = config["model"]["rnn_layers"],
        output_size = config["model"]["output_size"]
    )
    model.load_state_dict(torch.load(config["model"]["checkpoint"]))
    log.info("model weight {}".format(config["model"]["checkpoint"]))
    
    backtest_engine = BackTest(config = config,
                                model = model,
                                dataset=train_dataset)
    backtest_engine.run(log=log)
    
    
if __name__ == "__main__":
    main()