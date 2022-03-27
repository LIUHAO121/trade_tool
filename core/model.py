import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)
        # return output


class LSTMPred(nn.Module):
    def __init__(self,embed_dim,hidden_size,rnn_layers,output_size):
        super(LSTMPred, self ).__init__() 
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.output_size = output_size
        self.lstm1 = nn.LSTM(self.embed_dim,self.hidden_size,self.rnn_layers)
        self.ln1 = LayerNorm(self.hidden_size)
        self.relu1 = nn.ReLU(inplace=True)
        
        # self.lstm2 = nn.LSTM(self.hidden_size,self.hidden_size)
        # self.ln2 = LayerNorm(self.hidden_size)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.lstm3 = nn.LSTM(self.hidden_size,self.hidden_size)
        # self.ln3 = LayerNorm(self.hidden_size)
        # self.relu3 = nn.ReLU(inplace=True)
       
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        # self._init_weights()
        
    def _init_weights(self, scope=1.):
        self.fc1.weight.data.uniform_(-scope, scope)
        self.fc1.bias.data.fill_(0)
        
    def forward(self,x):
        """
        Args:
            x (Tensor):x.shape = (batch,seq_len,emb_dim)
        """
        x ,(h_n, c_n)= self.lstm1(x.transpose(0,1))
        x = self.ln1(x)
        # x, (h_n, c_n) = self.lstm2(x)
        # x = self.ln2(x)
       
        x = x[-1] 
        out = self.fc1(x)
        return out
        
         
if __name__ == "__main__":
    batch = 32
    embed_dim=4
    hidden_size=12
    rnn_layers=3
    output_size=3
    seq_len=5
    model = LSTMPred(embed_dim=embed_dim,hidden_size=hidden_size,rnn_layers=rnn_layers,output_size=output_size)
    input = torch.ones(batch,seq_len,embed_dim)

    output = model(input)
    print(output.shape)
        