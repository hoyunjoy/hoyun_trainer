import pdb
import torch
import torch.nn as nn

class Trainer():
    
    def __init__(self, model, optimizer, criterion, scheduler, train_loader, lr, weight_decay, **kwargs):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.lr = lr
        self.weight_decay = weight_decay
        
    
    def train(self, train_loader):
        
        self.model.train()
        
        for (inputs, targets) in train_loader:
            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss = self.criterion(pred, targets)
            loss.backward()
            self.optimizer.step()
            
            print("train loss: {.6f}".format(loss))
            
            ## if scheduler updates per batch
            if self.scheduler.per_batch == True:
                self.scheduler.step()
        
        if self.scheduler.per_batch == False:
            self.scheduler.step()

    def evaluate(self, test_loader):
        
        self.model.eval()
        
        with torch.no_grad():
            for (inputs, targets) in test_loader:
                
                pred = self.model(inputs)
                loss = self.criterion(pred, targets)
                
                print("test loss: {.6f}".format(loss))