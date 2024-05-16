import pdb
import torch
import torch.nn as nn
import importlib

class Network():

    def __init__(self, model, loss, optimizer, scheduler, **kwargs):
            
        ## Define model
        self.__M__ = importlib.import_module('model.' + model).MainModel(**kwargs)
        
        ## Define loss function
        self.__L__ = importlib.import_module('loss.' + loss).LossFunction(**kwargs)
        
        ## Define optimizer
        self.__Optimizer__ = importlib.import_module('optimizer.' + optimizer).Optimizer(self.__M__.parameters(), **kwargs)
        
        ## Define scheduler
        self.__Scheduler__ = importlib.import_module('scheduler.' + scheduler).Scheduler(self.__Optimizer__, **kwargs)
            
    
    def forward(self, inputs):
        outputs = self.__M__(inputs)
        loss = self.__L__(outputs)
        
        return loss

class Trainer():
    
    def __init__(self, network, train_loader, test_loader, **kwargs):
        
        self.network = network
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    
    def train(self):
        
        self.network.__M__.train()
        
        total_loss = 0
        
        for (inputs, labels, lengths_of_inputs, lengths_of_labels) in self.train_loader:
        
            self.network.__Optimizer__.zero_grad()
            pred = self.network.__M__(inputs)
            loss = self.network.__L__(pred, labels, lengths_of_inputs, lengths_of_labels)
            total_loss += loss
            loss.backward()
            self.network.__Optimizer__.step()
            
            ## if scheduler updates per batch
            if self.scheduler.lr_step == 'batch':
                self.scheduler.step()
        
        total_loss /= len(self.train_loader)
        
        print("train loss: {.6f}".format(total_loss))
        
        if self.scheduler.lr_step == 'epoch':
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