import torch

class LossFunction():
    
    def __init__(self, **kwargs):
        
        super(LossFunction, self).__init__()
        
        self.criterion = torch.nn.CTCLoss(blank=0)
        
        print('Initialized CTC Loss')

    def forward(self, pred, label, length_of_inputs, length_of_labels):
        
        nloss = self.criterion(pred, label, length_of_inputs, length_of_labels)
        
        return nloss