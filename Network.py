import pdb
import torch
import torch.nn as nn
import importlib

class Network(nn.Module):

    def __init__(self, model, loss, optimizer, scheduler, **kwargs):
        
        super(Network, self).__init__()
            
        ## Define model
        self.__M__ = importlib.import_module('model.' + model).MainModel(**kwargs)
        
        ## Define loss function
        self.__L__ = importlib.import_module('loss.' + loss).LossFunction(**kwargs)
        
        ## Define optimizer
        self.__Optimizer__ = importlib.import_module('optimizer.' + optimizer).Optimizer(self.__M__.parameters(), **kwargs)
        
        ## Define scheduler
        self.__Scheduler__ = importlib.import_module('scheduler.' + scheduler).Scheduler(self.__Optimizer__, **kwargs)
            
    
    def forward(self, inputs, labels, lengths_of_inputs, lengths_of_labels):
        preds = self.__M__(inputs)
        loss = self.__L__(preds, labels, lengths_of_inputs, lengths_of_labels)
        
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
            
            ## Move tensors to CUDA
            inputs = inputs.cuda()
            labels = labels.cuda()
            lengths_of_inputs = lengths_of_inputs.cuda()
            lengths_of_labels = lengths_of_labels.cuda()
        
            self.network.__Optimizer__.zero_grad()
            loss = self.network(inputs, labels, lengths_of_inputs, lengths_of_labels)
            total_loss += loss.item()
            loss.backward()
            self.network.__Optimizer__.step()
            
            ## if scheduler updates per batch
            if self.network.__Scheduler__[1] == 'batch':
                self.network.__Scheduler__[0].step()
        
        total_loss /= len(self.train_loader)
        
        print("train loss: {:.6f}".format(total_loss))
        
        if self.network.__Scheduler__[1] == 'epoch':
            self.network.__Scheduler__[0].step()

    def evaluate(self):
        
        self.model.__M__.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for (inputs, labels, lengths_of_inputs, lengths_of_labels) in self.test_loader:
                
                ## Move tensors to CUDA
                inputs = inputs.cuda()
                labels = labels.cuda()
                lengths_of_inputs = lengths_of_inputs.cuda()
                lengths_of_labels = lengths_of_labels.cuda()
                
                loss = self.network(inputs, labels, lengths_of_inputs, lengths_of_labels)
                total_loss += loss.item()
            
        total_loss /= len(self.test_loader)
        
        print("test loss: {:.6f}".format(total_loss))