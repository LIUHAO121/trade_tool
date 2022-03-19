import torch
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

def main():
    args = parse_args()
    config = load_json(args.cfg)
    
    train_dataset = StockDataSet(
        data_dir = config["dataset"]["train_dir"],
        coloumns = config["dataset"]["coloumns"],
        seq_len =  config["dataset"]["seq_len"],
        pred_len = config["dataset"]["pred_len"]
                           )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size = config["dataset"]["batch_size"],
        shuffle = True,
        )
    
    for x, y in train_loader:
        print(x.shape)
        
        
if __name__ == "__main__":
    main()
