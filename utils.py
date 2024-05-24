import torch

char_list = [' ', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z']

## Put blank in the first place
char_list.insert(0, '-')

## Make dictionary between characters and indices
idx2char = {i:char for i, char in enumerate(char_list)}
char2idx = {char:i for i, char in enumerate(char_list)}

## Convert all letters to corresponding integers
def strToInt(sentence, char2idx):
    
    int_list = []
    
    for letter in sentence:
        int_list.append(char2idx[letter])
    
    return torch.IntTensor(int_list)

## Convert all integers to corresponding letters
def intToStr(int_tensor, idx2char):
    
    char_str = ""
    
    for integer in int_tensor:
        char_str += idx2char[integer.item()]

    return char_str