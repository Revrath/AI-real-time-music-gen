import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class MusicDataset(Dataset):
    def __init__(self, data_path, sequence_length):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.notes = data['notes']
        self.labels = data['labels']
        self.seq_len = sequence_length
        
        self.vocab = sorted(list(set(self.notes)))
        self.vocab_size = len(self.vocab)
        self.note_to_int = {n: i for i, n in enumerate(self.vocab)}
        self.int_to_note = {i: n for i, n in enumerate(self.vocab)}
        
        self.data_indices = [self.note_to_int[n] for n in self.notes]
        
    def __len__(self):
        return len(self.data_indices) - self.seq_len

    def __getitem__(self, idx):
        # Input (x): notes from 0 to 99
        # Target (y): notes from 1 to 100 
        
        chunk = self.data_indices[idx : idx + self.seq_len + 1]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        # one label per sequence    
        label_val = self.labels[idx + self.seq_len]
        label = torch.tensor(label_val, dtype=torch.float)
        
        return x, y, label