import pdb
import torch
import torch.nn as nn
import torchaudio
import argparse
from DatasetLoader import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Hoyun's Trainer")

parser.add_argument('--config',     type=str,   default=None,   help='Config YAML file')

## Dataset
parser.add_argument('--train_path',   type=str, default="/mnt/lynx3/datasets/LibriSpeech_wav/train-clean-100/")
parser.add_argument('--val_path',     type=str, default="/mnt/lynx3/datasets/LibriSpeech_wav/dev-clean")
parser.add_argumnet('--test_path',    type=str, default="/mnt/lynx3/datasets/LibriSpeech_wav/test-clean")
parser.add_argument('--sample_rate',  type=str, default=16000)
parser.add_argument('--ext',          type=str, default='wav')

## Preprocessing
parser.add_argument('--mel',        dest='mel', action='store_true')
parser.add_argument('--n_fft',      type=int,   default=256)

## Data loader
parser.add_argument('--batch_size',     type=int,   default=64)
parser.add_argument('--seed',           type=int,   default=42)
parser.add_argument('--num_workers',    type=int,   default=5)

## Training Details
parser.add_argument('--test_interval',  type=int,   default=5)
parser.add_argument('--max_epoch',      type=int,   default=100)
parser.add_argument('--loss',           type=str,   default='mse')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default='adam')
parser.add_argument('--scheduler',      type=str,   default='steplr')
parser.add_argument('--lr',             type=float, default=0.001)
parser.add_argument('--lr_decay',       type=float, default=0.9)
parser.add_argument('--weight_decay',   type=float, default=0)

## Load and save
parser.add_argument('--initial_model',  type=str,   default="")
parser.add_argument('--save_path',      type=str)

## Model
parser.add_argument('--model',  type=str)

## For test only
parser.add_argument('--eval',   dest='eval',    action='store_true')

## Distributed
parser.add_argument('--distributed',    dest='distributed', action='store_true')

## port
parser.add_argument('--port',   type=str,   default='8888')

args = parser.parse_args()

def main():
    
    ## Fix seed
    torch.manual_seed(args.seed)
    
    ## Define datasets
    train_dataset = train_dataset(vars(args))
    test_dataset = test_dataset(vars(args))
    
    ## Define data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )
    
    ## When training, test loader is val loader
    ## When evaluating, test loader is test loader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # sampler=test_sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )
    
    ## Define Model
    model = MainModel
    
    ## Define Trainer
    trainer = model.trainer
    
    
    
    
    

if __name__ == "__main__":
    main()