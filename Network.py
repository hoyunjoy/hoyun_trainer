import pdb
import torch
import torch.nn as nn
import importlib

# class Network():
    
#     def __init__(self, **kwargs):
#         self.__M__ = importlib.import_module(model).MainModel()
#         self.__L__ = importlib.import_module(loss_function).LossFunction()
#         self.__Optimizer__ = importlib.import_module(optimizer).Optimizer()
#         self.__Scheduler__ = importlib.import_module(scheduler).Scheduler()
    
#     def forward(self, inputs):
#         outputs = self.__M__(inputs)
#         loss = self.__L__(outputs)

class Trainer():
    
    def __init__(self, model, optimizer, criterion, scheduler, train_loader, test_loader, lr, weight_decay, **kwargs):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.weight_decay = weight_decay
        
    
    def train(self):
        
        self.model.train()
            
        total_loss = 0
            
        for (inputs, labels, lengths_of_inputs, lengths_of_labels) in self.train_loader:
        
            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss = self.criterion(pred, targets)
            total_loss += loss
            loss.backward()
            self.optimizer.step()
            
            ## if scheduler updates per batch
            if self.scheduler.per_batch:
                self.scheduler.step()
        
        total_loss /= len(self.train_loader)
        
        print("train loss: {.6f}".format(loss))
        
        if not self.scheduler.per_batch:
            self.scheduler.step()

    def evaluate(self):
        
        self.model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for (inputs, targets) in self.test_loader:
                
                pred = self.model(inputs)
                total_loss += self.criterion(pred, targets)
            
            total_loss /= len(self.test_loader)
            print("test loss: {.6f}".format(total_loss))