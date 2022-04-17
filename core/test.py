import argparse
import os
import logging
import pandas as pd
import datetime as dt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from data_tool import get_qfq
from core import LSTMPred, StockClsDataSet ,StockRegDataSet ,load_json, setup_logging, get_current_logger,plot_test_out,plot_multi_test_out



"""
用全部数据训练，并预测未来结果
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/reg_config_close.json",
                        type=str)
    args = parser.parse_args()

    return args


LOSS_FUN_LIB = {
            "classification":nn.CrossEntropyLoss(),
            "regression":nn.MSELoss()
               }

DATASET_LIB = {
    "classification": StockClsDataSet,
    "regression":StockRegDataSet
}

OPTIMIZER_LIB = {
    "sgd":torch.optim.SGD,
    "adam":torch.optim.Adam
    
}

def train(dataloader, model, loss_fn, optimizer, log, config):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if config["task_type"] == "classification":
                batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item() / X.shape[0]
                log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] batch Accuracy:{(100*batch_correct):>0.1f}%")
            else:
                log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] ")
 
 
def company_train_and_predict(company_name, end_date):
    stock_basic = pd.read_csv("data/stock_basic.csv") 
    ts_code = stock_basic[stock_basic["name"]==company_name]["ts_code"].values[0]
    list_date = stock_basic[stock_basic["name"]==company_name]["list_date"].values[0]
    
    # 2 删除旧数据，获取新股票数据并保存
    data_dir = "data/test_train"
    for csv_file in os.listdir(data_dir):
        csv_path = os.path.join(data_dir,csv_file)
        os.remove(csv_path)
    train_df = get_qfq(ts_code = ts_code, start_date=str(list_date), end_date=end_date)
    train_df.to_csv("{}/{}_{}.csv".format(data_dir,ts_code,end_date))
    
    # 3 训练并保存模型
    args = parse_args()
    config = load_json(args.cfg)
    task_type = config["task_type"]
    

    log = logging.getLogger(__name__)
    log.setLevel(level = logging.INFO)
    handler = logging.FileHandler("{}/log/{}_{}.txt".format(config["log_dir"],company_name, dt.datetime.now().strftime('%Y%m%d-%H%M%S')))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
   
    for key in config:
        log.info(key)
        log.info(config[key])
 
    dataset_task = DATASET_LIB[task_type]
    train_dataset = dataset_task(
        dataset_type = "train",
        data_dir = "data/test_train",
        coloumns = config["dataset"]["coloumns"],
        seq_len =  config["dataset"]["seq_len"],
        pred_len = config["dataset"]["pred_len"],
        split= 1.0)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size = config["dataset"]["batch_size"],
        shuffle = config["dataset"]["shuffle"],
        )
    
    model = LSTMPred(
        embed_dim = config["model"]["embed_dim"],
        hidden_size = config["model"]["hidden_size"],
        rnn_layers = config["model"]["rnn_layers"],
        output_size = config["model"]["output_size"]
    )
    
    model.to(device)
    weight_dir = config["model"]["weight_dir"]
    
    
    loss_fn = LOSS_FUN_LIB[task_type]
    optimizer = OPTIMIZER_LIB[config["optimizer"]["name"]](model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9)
    
    num_epoch = config["epoch"]

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer,
        T_max = num_epoch
        )
    if not config["model"]["checkpoint"]:
        for t in range(num_epoch):
            log.info(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, log, config)
            scheduler.step()
        torch.save(model.state_dict(), os.path.join(weight_dir,f'{ts_code}_e{num_epoch}.pth'))
    else:
        log.info("resume model from {} ...".format(config["model"]["checkpoint"]))
        model.load_state_dict(torch.load(config["model"]["checkpoint"]))
    
    # 4 预测未来走势并可视化
 
    interval = config["plot_interval"]
    past_real_values, future_predicted_multi = train_dataset.test_predict_dense(model=model,
                                                                                interval=interval,
                                                                                num_interval=30)
    plot_multi_test_out(predicted_datas=future_predicted_multi,
                        real_values=past_real_values,
                        interval=interval,
                        model_tag="{}_future_predict_dense".format(company_name))
 
def main():
    end_date = '20220417'
    # 1 获取股票编码和上市日期
    company_names = ["科大讯飞","海康威视"]
    for name in company_names:
        print("training {}".format(name))
        company_train_and_predict(company_name=name, end_date=end_date)
    
    
        
if __name__ == "__main__":
    main()