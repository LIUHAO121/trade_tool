import os
import datetime as dt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
from core import LSTMPred, StockClsDataSet ,StockRegDataSet ,load_json,setup_logging,get_current_logger,plot_results_multiple,plot_results_point_by_point




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
            
      
def test(dataloader, model, loss_fn,log, config):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if config["task_type"] == "classification":
                correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
    test_loss /= num_batches
    if config["task_type"] == "classification":
        correct /= size
        log.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")     
    else:
       log.info(f"Test Error: \n Avg loss: {test_loss:>8f} \n")   

def main():
    args = parse_args()
    config = load_json(args.cfg)
    task_type = config["task_type"]
    
    setup_logging(
        log_dir = config["log_dir"],
        log_level = "INFO",
        trace_id=dt.datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
    log = get_current_logger()
    for key in config:
        log.info(key)
        log.info(config[key])
 
    dataset_task = DATASET_LIB[task_type]
    train_dataset = dataset_task(
        dataset_type = "train",
        data_dir = config["dataset"]["train_dir"],
        coloumns = config["dataset"]["coloumns"],
        seq_len =  config["dataset"]["seq_len"],
        pred_len = config["dataset"]["pred_len"],
        split= config["dataset"]["train_test_split"]
                           )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size = config["dataset"]["batch_size"],
        shuffle = False,
        )
    
    test_dataset = dataset_task(
        dataset_type = "test",
        data_dir = config["dataset"]["test_dir"],
        coloumns = config["dataset"]["coloumns"],
        seq_len =  config["dataset"]["seq_len"],
        pred_len = config["dataset"]["pred_len"],
        split = config["dataset"]["train_test_split"]
                           )
  
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size = config["dataset"]["batch_size"],
        shuffle = True,
        )
    
    model = LSTMPred(
        embed_dim = config["model"]["embed_dim"],
        hidden_size = config["model"]["hidden_size"],
        rnn_layers = config["model"]["rnn_layers"],
        output_size = config["model"]["output_size"]
    )
    resume = config["model"]["resume"]
    if resume:
        log.info("resume model from {} ...".format(config["model"]["weight"]))
        model.load_state_dict(torch.load(config["model"]["weight"]))
    model.to(device)
    weight_dir = config["model"]["weight_dir"]
    
    
    loss_fn = LOSS_FUN_LIB[task_type]
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(),lr=config["optimizer"]["lr"])
    
    num_epoch = config["epoch"]
    # 更新学习率
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer,
        T_max = num_epoch
        )

    for t in range(num_epoch):
        log.info(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, log, config)
        test(test_dataloader, model, loss_fn, log, config)
        scheduler.step()
        # torch.save(model.state_dict(), os.path.join(weight_dir,'model_e{}.pth'.format(t)))
        
    gts, preds = test_dataset.predict_sequences_multiple(model)
    plot_results_multiple(preds,gts,seq_len=config["dataset"]["seq_len"],model_tag="seq_multiple")
    
    gts, preds = test_dataset.predict_point_by_point(model)
    plot_results_point_by_point(preds, gts, seq_len=config["dataset"]["seq_len"], model_tag="point_by_point")
    log.info("Done!")
    
        
    
        
        
if __name__ == "__main__":
    main()