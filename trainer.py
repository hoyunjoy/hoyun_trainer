import pdb
import torch
import torch.nn as nn
import torchaudio
import importlib
import argparse
from DatasetLoader import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from Network import *
from utils import *

parser = argparse.ArgumentParser(description="Hoyun's Trainer")

parser.add_argument('--config',     type=str,   default=None,   help='Config YAML file')

## Dataset
parser.add_argument('--train_path',   type=str, default="/mnt/lynx3/datasets/LibriMix/LibriSpeech/train-clean-100/")
parser.add_argument('--val_path',     type=str, default="/mnt/lynx3/datasets/LibriMix/LibriSpeech/dev-clean/")
parser.add_argument('--test_path',    type=str, default="/mnt/lynx3/datasets/LibriMix/LibriSpeech/test-clean/")
parser.add_argument('--sample_rate',  type=str, default=16000)
parser.add_argument('--ext',          type=str, default='flac')

## Data transformation
parser.add_argument('--n_mels',     type=int,   default=128)
parser.add_argument('--n_fft',      type=int,   default=1024)
parser.add_argument('--win_length', type=int,   default=1024)
parser.add_argument('--hop_length', type=int,   default=512)
parser.add_argument('--transform',  type=str,   default='melspec')

## Data loader
parser.add_argument('--batch_size',     type=int,   default=64)
parser.add_argument('--seed',           type=int,   default=42)
parser.add_argument('--num_workers',    type=int,   default=5)

## Training Details
parser.add_argument('--test_interval',  type=int,   default=5)
parser.add_argument('--max_epoch',      type=int,   default=100)
parser.add_argument('--loss',           type=str,   default='ctc')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default='adam')
parser.add_argument('--scheduler',      type=str,   default='steplr')
parser.add_argument('--lr',             type=float, default=0.001)
parser.add_argument('--lr_decay',       type=float, default=0.9)
parser.add_argument('--weight_decay',   type=float, default=0)

## Load and save
parser.add_argument('--initial_model',  type=str,   default="./exps/exp1/model0015.pt")
parser.add_argument('--save_path',      type=str,   default="./exps/exp1/")

## Model
parser.add_argument('--model',  type=str,   default='SpeechRecognitionNet')

## For test only
parser.add_argument('--eval',   dest='eval',    action='store_true')

## Distributed
parser.add_argument('--distributed',    dest='distributed', action='store_true')

## port
parser.add_argument('--port',   type=str,   default='8888')

args = parser.parse_args()

## Pad inputs and labels
def collate_fn(batch):
    
    ## Pad inputs
    ## input shape of pad_sequence: [batch, time, *] if batch_first=True
    audio_before_pad = [audio.transpose(0, 1) for audio, utterance in batch]            # -> [batch, n_mels, time]
    audio_after_pad  = pad_sequence(audio_before_pad, batch_first=True).transpose(1, 2) # -> [batch, time, n_mels]
    
    lengths_of_audio = torch.IntTensor([len(audio) for audio in audio_before_pad])
    
    ## Pad labels
    utterance_before_pad = [strToInt(utterance, char2idx) for audio, utterance in batch]
    utterance_after_pad = pad_sequence(utterance_before_pad, batch_first=True)
    
    lengths_of_utterance = torch.IntTensor([len(utterance) for utterance in utterance_before_pad])
    
    return audio_after_pad, utterance_after_pad, lengths_of_audio, lengths_of_utterance


## Find all characters in dataset
def findAllChar(dataset):
    
    char_list = []
    
    for idx in range(len(dataset)):
        sentence = dataset[idx][1]
        for letter in sentence:
            
            if letter in char_list:
                continue
            else:
                char_list.append(letter)
    
    return sorted(char_list)



def main():
    
    ## Fix seed
    torch.manual_seed(args.seed)
    
    ## Define datasets
    train_set = train_dataset(**vars(args))
    test_set = test_dataset(**vars(args))
    
    # Execute this only at first
    # ## Find all characters
    # char_list = findAllChar(train_set)
    # char_list_2 = findAllChar(test_set)
    
    # assert char_list == char_list_2
        
    ## Define data loaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False,
        # sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    ## When training, test loader is val loader
    ## When evaluating, test loader is test loader
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        # sampler=test_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    speaker_recognition_net = Network(**vars(args)).cuda()
        
    ## Define Trainer
    trainer = Trainer(network=speaker_recognition_net,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      **vars(args))
    
    trainer.loadParameters()
    
    if args.eval:
        trainer.evaluate()
        return
    
    for epoch in range(1, args.max_epoch + 1):
        trainer.train()
        if epoch % args.test_interval == 0:
        
            trainer.evaluate()
            trainer.saveParameters(epoch)
    
    
    
    

if __name__ == "__main__":
    main()