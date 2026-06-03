# ------------------------------------------------------------------------------
# Written by MSP (alstjd135@gmail.com)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
                
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        
    def forward(self, x):
        '''
            x = (batch_size, seq_length, input_size)
            lstm_out = (batch_size, seq_length, hidden_size)
            h_out, c_out = (batch_size, num_layers, hidden_size)
            out = (batch_size*num_layers, hidden_size)
        '''
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        lstm_out, (h_out, c_out) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        
        return h_out
        
        

class LSTM_fc(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM_fc, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data)            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1, std=0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, x):
        '''
            x = (batch_size, seq_length, input_size)
            lstm_out = (batch_size, seq_length, hidden_size)
            h_out, c_out = (batch_size, num_layers, hidden_size)
            out = (batch_size*num_layers, num_classes)
        '''
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        lstm_out, (h_out, c_out) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
                
        out = self.fc(h_out)

        return out
    
    
    
if __name__ == "__main__":
    
    lstm = LSTM(
        input_size=256,
        hidden_size=64,
        num_layers=1
    )
    
    print("========== LSTM ==========")
    for idx, m in enumerate(lstm.modules()):
        print("{} : {}".format(idx, m))
    
    # x = (batch, seq_length, input_size)
    x = torch.rand(8, 16, 256)
    y = lstm(x)    
    print("x = {}".format(x.shape))
    print("y = {}".format(y.shape))
    
    lstm_fc = LSTM_fc(
        input_size=256,
        hidden_size=64,
        num_layers=1,
        num_classes=7       
    )
    print("==========================")
    print()    

    print("========== LSTM_fc ==========")
    for idx, m in enumerate(lstm_fc.modules()):
        print("{} : {}".format(idx, m))
    
    # x = (batch, seq_length, input_size)
    x = torch.rand(8, 16, 256)
    y = lstm_fc(x)    
    
    print("x = {}".format(x.shape))
    print("y = {}".format(y.shape))
    print("==============================")
    
    