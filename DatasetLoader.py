import pdb
import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

def waveToSpec(waveform, n_fft, win_length=None, hop_length=None):
    
    transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        )
    
    spec = transform(waveform)
    
    return spec

def waveToMelSpec(waveform, sample_rate, n_fft, n_mels, win_length=None, hop_length=None):
    
    transforms = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    melspec = transforms(waveform)
    
    return melspec



class train_dataset(Dataset):
    
    def __init__(self, train_path, ext, n_fft, n_mels, win_length, hop_length, transform=None, **kwargs):
        self.train_root = Path(train_path)
        self.audio_path_list = [audio_path for audio_path in self.train_root.glob('*/*/*.' + ext)]
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.transform = transform
        
    def __len__(self):
        return len(self.audio_path_list)
        
    def __getitem__(self, idx):
        parent = self.audio_path_list[idx].parent
        audio_file_name = self.audio_path_list[idx].name
        first, second, third = audio_file_name.split('-')
        
        ## Load text(GT)
        text_file_name = first + '-' + second + '.trans.txt'
        text_file_path = os.path.join(parent, text_file_name)
        
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            for line in text_file:
                audio_file_name_of_txt, utterance = line.rstrip('\n').split(' ', 1)
                if audio_file_name == audio_file_name_of_txt:
                    break
        
        ## Load audio
        audio_file_path = self.audio_path_list[idx]
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        ## Transform audio
        if self.transform=='spec':
            spec = waveToSpec(waveform=waveform.detach(),
                              n_fft=self.n_fft,
                              win_length=self.win_length,
                              hop_length=self.hop_length)
            return spec, utterance
        
        elif self.transform=='melspec':
            melspec = waveToMelSpec(waveform=waveform.detach(),
                                    sample_rate=sample_rate,
                                    n_fft=self.n_fft,
                                    n_mels=self.n_mels,
                                    win_length=self.win_length,
                                    hop_length=self.hop_length)
            return melspec.squeeze(0), utterance
        
        return waveform.detach(), utterance
        
class test_dataset(Dataset):
    
    def __init__(self, test_path, ext, n_fft, n_mels, win_length, hop_length, transform=None, **kwargs):
        self.test_root = Path(test_path)
        self.audio_path_list = [audio_path for audio_path in self.test_root.glob('*/*/*.' + ext)]
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.transform = transform
        
    def __len__(self):
        return len(self.audio_path_list)
        
    def __getitem__(self, idx):
        parent = self.audio_path_list[idx].parent
        audio_file_name = self.audio_path_list[idx].name
        first, second, third = audio_file_name.split('-')
        
        ## Load text(GT)
        text_file_name = first + '-' + second + '.trans.txt'
        text_file_path = os.path.join(parent, text_file_name)
        
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            for line in text_file:
                audio_file_name_of_txt, utterance = line.rstrip('\n').split(' ', 1)
                if audio_file_name == audio_file_name_of_txt:
                    break
        
        ## Load audio
        audio_file_path = self.audio_path_list[idx]
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        
        ## Transform audio
        if self.transform=='spec':
            spec = waveToSpec(waveform=waveform,
                              n_fft=self.n_fft,
                              win_length=self.win_length,
                              hop_length=self.hop_length)
            return spec.squeeze(0), utterance
        
        elif self.transform=='melspec':
            melspec = waveToMelSpec(waveform=waveform,
                                    sample_rate=sample_rate,
                                    n_fft=self.n_fft,
                                    n_mels=self.n_mels,
                                    win_length=self.win_length,
                                    hop_length=self.hop_length)
            return melspec, utterance
        
        return waveform, utterance