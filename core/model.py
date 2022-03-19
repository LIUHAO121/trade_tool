import torch
import torch.nn as nn
import torch.nn.functional as F




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


class LSTMPred(nn.Module):
    def __init__(self,embed_dim,hidden_size,rnn_layers,output_size):
        super(LSTMPred, self ).__init__() 
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(self.embed_dim,self.hidden_size,self.rnn_layers)
        self.ln = LayerNorm(self.hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(self. hidden_size,64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
        
       
        
    def forward(self,x):
        """

        Args:
            x (Tensor):x.shape = (batch,seq_len,emb_dim)
        """
        x ,(h_n, c_n)= self.lstm(x.transpose(0,1))
        print(x.shape)
        print(h_n.shape)
        x = self.ln(x)
        x = x[-1] 
        out = self.fc(x)
        return out
        
         
if __name__ == "__main__":
    batch = 32
    embed_dim=4
    hidden_size=12
    rnn_layers=3
    output_size=3
    seq_len=5
    model = LSTMPred(embed_dim=embed_dim,hidden_size=hidden_size,rnn_layers=rnn_layers,output_size=output_size)
    input = torch.ones(seq_len,batch,embed_dim)

    output = model(input)
    # print(output)
        