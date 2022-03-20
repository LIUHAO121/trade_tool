import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from core import LSTMPred, StockDataSet ,load_json,setup_logging,get_current_logger
import datetime as dt

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    args = parser.parse_args()

    return args


def train(dataloader, model, loss_fn, optimizer,log):
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

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item() / X.shape[0]
            
            log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] batch Accuracy:{(100*batch_correct):>0.1f}%")
            
      
def test(dataloader, model, loss_fn,log):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    log.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")     

def main():
    args = parse_args()
    config = load_json(args.cfg)
    
    setup_logging(
        log_dir = config["log_dir"],
        log_level = "INFO",
        trace_id=dt.datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
    log = get_current_logger()
    for key in config:
        log.info(key)
        log.info(config[key])
 
    
    train_dataset = StockDataSet(
        name = "train",
        data_dir = config["dataset"]["train_dir"],
        coloumns = config["dataset"]["coloumns"],
        seq_len =  config["dataset"]["seq_len"],
        pred_len = config["dataset"]["pred_len"]
                           )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size = config["dataset"]["batch_size"],
        shuffle = False,
        )
    
    test_dataset = StockDataSet(
        name = "test",
        data_dir = config["dataset"]["test_dir"],
        coloumns = config["dataset"]["coloumns"],
        seq_len =  config["dataset"]["seq_len"],
        pred_len = config["dataset"]["pred_len"]
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
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9)
    
 
    epochs = config["epoch"]
    for t in range(epochs):
        log.info(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer,log)
        test(test_dataloader, model, loss_fn,log)
    log.info("Done!")
        
    
        
        
if __name__ == "__main__":
    main()