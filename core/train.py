import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from core import LSTMPred, StockDataSet ,load_json


device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    
    

    args = parser.parse_args()

    return args


def train(dataloader, model, loss_fn, optimizer):
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
            batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item() / X.shape[0]
            
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] batch Accuracy:{(100*batch_correct):>0.1f}%")
            
      
def test(dataloader, model, loss_fn):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")     
     
def main():
    args = parse_args()
    config = load_json(args.cfg)
    
    train_dataset = StockDataSet(
        data_dir = config["dataset"]["train_dir"],
        coloumns = config["dataset"]["coloumns"],
        seq_len =  config["dataset"]["seq_len"],
        pred_len = config["dataset"]["pred_len"]
                           )
    train_dataset.info()
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size = config["dataset"]["batch_size"],
        shuffle = True,
        )
    
    test_dataloader = DataLoader(
        dataset=train_dataset,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    epochs = config["epoch"]
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
        
    
        
        
if __name__ == "__main__":
    main()