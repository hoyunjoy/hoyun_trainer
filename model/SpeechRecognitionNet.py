import pdb
import torch
import torch.nn as nn


class SpeechRecognitionNet(nn.Module):
    
    def __init__(self, n_mels, out_channels=29, **kwargs):
        
        super(SpeechRecognitionNet, self).__init__()
        
        self.sequential = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(n_mels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        self.fc = nn.Linear(64*2, out_channels) # *2 due to bidirectional
        
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, inputs):
        
        # inputs.shape: [batch, n_mels, time]
        
        x = self.sequential(inputs)         # -> [batch, n_mels, time]
        batch, n_mels, time = x.shape
        x = x.permute(0, 2, 1)              # -> [batch, time, n_mels]
        x, _ = self.lstm(x)                 # -> [batch, time, n_mels]
        x = self.fc(x)                      # -> [batch, time, 29]
        outputs = self.log_softmax(x)       # -> [batch, time, 29]
        ## Change outputs shape to use CTCLoss
        outputs = outputs.permute(1, 0, 2)  # -> [time, batch, 29]
        
        return outputs

def MainModel(**kwargs):
    
    model = SpeechRecognitionNet(**kwargs)
    
    return model