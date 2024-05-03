import pdb
import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class train_dataset(Dataset):
    
    def __init__(self, train_path, ext, **kwargs):
        self.train_root = Path(train_path)
        self.audio_list = [train_audio for train_audio in self.train_root.glob('*/*/*.' + ext)]
        
    def __len__(self):
        return len(self.audio_list)
        
    def __getitem__(self, idx):
        audio = os.path.join(self.train_root, self.audio_list[idx])
        return audio
        
class test_dataset(Dataset):
    
    def __init__(self, test_path, **kwargs):
        
    def __len__(self):
        
    def __getitem__(self):