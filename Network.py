import os
import pdb
import torch
import torch.nn as nn
import importlib
from utils import *
from tqdm import tqdm
from trainer import intToStr, idx2char



## Calculate used length for ctc calcuation
def usedLengthForCtcCal(ratio, lengths_of_audio):
    
    used_length_for_ctc_cal = ratio * lengths_of_audio
    used_length_for_ctc_cal = torch.ceil(used_length_for_ctc_cal)
    used_length_for_ctc_cal = used_length_for_ctc_cal.type(torch.int).cuda()
    
    return used_length_for_ctc_cal

def removeBlanks(sentence):
    new_sentence = ""
    for letter in sentence:
        if letter == "-":
            continue
        else:
            new_sentence += letter
    
    return new_sentence

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
        
        preds = self.__M__(inputs)                                              # -> [time, batch, 29]
        loss = self.__L__(preds, labels, lengths_of_inputs, lengths_of_labels)
        
        return loss

class Trainer():
    
    def __init__(self, network, train_loader, test_loader, initial_model, save_path, **kwargs):
        
        self.network        = network
        self.train_loader   = train_loader
        self.test_loader    = test_loader
        self.save_path      = save_path
        self.initial_model  = initial_model 
        
    
    def train(self):
        
        self.network.__M__.train()
        
        total_loss = 0
        
        for (inputs, labels, lengths_of_inputs, lengths_of_labels) in tqdm(self.train_loader):
            
            # inputs.shape: [batch, n_mels, time]
            # labels.shape: [batch, utterence_length]
            # preds.shape:  [preds_lengths, batch, num_classes]
            
            ## Move tensors to CUDA
            inputs = inputs.cuda()
            labels = labels.cuda()
            lengths_of_inputs = lengths_of_inputs.cuda()
            lengths_of_labels = lengths_of_labels.cuda()
        
            self.network.__Optimizer__.zero_grad()
            preds = self.network.__M__(inputs) # -> [time, batch, 29]
            
            ## Calculate ratio of preds_lengths to inputs_pad_lengths
            inputs_pad_lengths = inputs.size(2)         # 517
            preds_lengths = preds.size(0)               # 511
            ratio = preds_lengths / inputs_pad_lengths  # 0.988
            
            ## Calculate predicted preds_lengths
            predicted_preds_lengths = usedLengthForCtcCal(ratio, lengths_of_inputs) # 452
            
            loss = self.network.__L__(preds, labels, predicted_preds_lengths, lengths_of_labels)
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
        
        self.network.__M__.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for (inputs, labels, lengths_of_inputs, lengths_of_labels) in tqdm(self.test_loader):
                
                # inputs.shape: [batch, n_mels, time]
                # labels.shape: [batch, utterence_length]
                # preds.shape:  [preds_lengths, batch, num_classes]
                
                ## Move tensors to CUDA
                inputs = inputs.cuda()
                labels = labels.cuda()
                lengths_of_inputs = lengths_of_inputs.cuda()
                lengths_of_labels = lengths_of_labels.cuda()
                
                preds = self.network.__M__(inputs)  # -> [time, batch, 29]
                
                ## Calculate ratio of preds_lengths to inputs_pad_lengths
                inputs_pad_lengths = inputs.size(2)
                preds_lengths = preds.size(0)
                ratio = preds_lengths / inputs_pad_lengths
                
                ## Calculate predicted preds_lengths
                predicted_preds_lengths = usedLengthForCtcCal(ratio, lengths_of_inputs)
                
                loss = self.network.__L__(preds, labels, predicted_preds_lengths, lengths_of_labels)
                total_loss += loss.item()
                
                ## Process preds to print results
                preds = preds.permute(1, 0, 2)      # -> [batch, time, 29]
                preds = torch.argmax(preds, dim=2)  # -> [batch, time]
                
                predicted_sentence = []
                for ii in range(preds.size(0)):
                    predicted_sentence.append(removeBlanks(intToStr(preds[ii], idx2char)))

        total_loss /= len(self.test_loader)
        print("test loss: {:.6f}".format(total_loss))
        
        print("label     : {}".format(removeBlanks(intToStr(labels[-1], idx2char))))
        print("prediction: {}".format(predicted_sentence[-1]))

        
    def saveParameters(self, epoch):
        
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.network.__M__.state_dict(), self.save_path + f"model{epoch:03d}.pt")
        
    def loadParameters(self):
        self.network.__M__.load_state_dict(torch.load(self.initial_model))
        print("{} is successively loaded.".format(self.initial_model))