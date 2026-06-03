# ------------------------------------------------------------------------------
# Written by MSP (alstjd135@gmail.com)
# ------------------------------------------------------------------------------

from models.backbone.lstm import LSTM
#from backbone.lstm import LSTM

import torch
import torch.nn as nn
import torchvision


class LSTM_NIA(nn.Module):
    def __init__(self, hidden_size=256, num_classes=7, num_layers=1, backbone='resnet50', device='cpu'):
        super(LSTM_NIA, self).__init__()
        self.input_size = 512           # lstm input size (resnet output)
        self.device = device
        backbone = torchvision.models.resnet50(pretrained=False)
        
        # remove last layer
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))      
        
        self.lstm = LSTM(
            input_size = self.input_size,
            hidden_size = hidden_size,
            num_layers = num_layers
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
        '''
        self.lstm_fc = LSTM_fc(
            num_classes = num_classes,
            input_size = 512,
            hidden_size = hidden_size,
            num_layers=1
        )
        '''
        self._to_device()
        
    def _to_device(self):
        for p in self.backbone.parameters():
            p = p.to(self.device)
        for p in self.lstm.parameters():
            p = p.to(self.device)
        for p in self.fc.parameters():
            p = p.to(self.device)
            
    def _check_device(self, module):
        for m in module.parameters():
            print(m.is_cuda)

    def forward(self, x):
        # x = (batch_size, seq_length, C, H, W)
        batch_size, img_size = x.shape[0], x.shape[2:]
        # merge batch_size and seq_length in order to feed everything to the cnn
        x = x.reshape(-1, *img_size)        
        # x : (batch_size*seq_length, C, H, W) → (batch_size, seq_length, input_size)        
        x = self.backbone(x)
        x = x.reshape(batch_size, -1, self.input_size)
        
        # lstm : (batch_size, seq_length, input_size) → (batch_size, num_classes)
        out = self.lstm(x)
        out = self.fc(out)
        
        return out


if __name__ == "__main__":
    
    lstm_nia = LSTM_NIA()
    
    batch_size = 2
    seq_length = 6
    
    x = torch.rand(batch_size, seq_length, 3, 224, 224)
    y = lstm_nia(x)
    print("x = {}".format(x.shape))
    print("y = {}".format(y.shape))
        
    
    