import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class MusicDataset(Dataset):
    def __init__(self, data_path, sequence_length):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.notes = data['notes']
        self.durations = data['durations']
        self.labels = data['labels']
        self.seq_len = sequence_length
        
        self.note_vocab = sorted(list(set(self.notes)))
        self.vocab_size = len(self.note_vocab)

        self.note_to_int = {n: i for i, n in enumerate(self.note_vocab)}
        self.int_to_note = {i: n for i, n in enumerate(self.note_vocab)}
        
        self.dur_vocab = sorted(list(set(self.durations)))
        self.dur_to_int = {d: i for i, d in enumerate(self.dur_vocab)}

        self.note_indices = [self.note_to_int[n] for n in self.notes]
        self.dur_indices = [self.dur_to_int[d] for d in self.durations]
        
    def __len__(self):
        return len(self.note_indices) - self.seq_len

    def __getitem__(self, idx):
        # Input (x): notes from 0 to 99
        # Target (y): notes from 1 to 100 
        
        note_chunk = self.note_indices[idx : idx + self.seq_len + 1]
        dur_chunk = self.dur_indices[idx : idx + self.seq_len + 1]

        x_note = torch.tensor(note_chunk[:-1], dtype=torch.long)
        x_dur = torch.tensor(dur_chunk[:-1], dtype=torch.long)
        
        y_note = torch.tensor(note_chunk[1:], dtype=torch.long)
        y_dur = torch.tensor(dur_chunk[1:], dtype=torch.long)

        # one label per sequence    
        label_val = self.labels[idx + self.seq_len]
        label = torch.tensor(label_val, dtype=torch.float)
        
        return x_note, x_dur, y_note, y_dur, label