import torch
import torch.nn as nn

class LossFunction(nn.Module):
    
    def __init__(self, **kwargs):
        
        super(LossFunction, self).__init__()
        
        self.criterion = torch.nn.CTCLoss(blank=0)
        
        print('Initialized CTC Loss')

    def forward(self, preds, labels, lengths_of_inputs, lengths_of_labels):

        nloss = self.criterion(preds, labels, lengths_of_inputs, lengths_of_labels)
        
        return nloss