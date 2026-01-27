import torch
import torch.nn as nn
import math

class MusicTransformer(nn.Module):
    def __init__(self, note_vocab_size, dur_vocab_size, embed_dim=128, num_heads=2, num_layers=2, seq_len=100, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        
        # change notes into matrix (vocab_size x embed_dim) descriptions (embeddings)
        # during training, these embeddings will be adjusted to musical patterns
        self.note_embedding = nn.Embedding(note_vocab_size, embed_dim)
        self.dur_embedding = nn.Embedding(dur_vocab_size, embed_dim)

        # embedding for notes positions (note at the beginning is different than note at the end)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        
        # projecting danger level to higher dimension
        self.danger_projection = nn.Linear(1, embed_dim)

        # feedforward layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        # stack num_layers of the above encoder_layer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # fully connected output layer to predict the next note
        self.fc_note = nn.Linear(embed_dim, note_vocab_size)
        self.fc_dur = nn.Linear(embed_dim, dur_vocab_size)
        
        # danger level?
         

    def forward(self, x_note, x_dur, danger_level=None):
        B, T = x_note.shape        
        positions = torch.arange(0, T, device=x_note.device).unsqueeze(0)
        
        # input is sum of note description, duration and position 
        # available space in this matrix is big enough to hold both types of information
        x = self.note_embedding(x_note) + self.dur_embedding(x_dur) + self.position_embedding(positions)
        
        if danger_level is not None:
            # [Batch] to [Batch, 1, 1]
            d = danger_level.view(B, 1, 1).float()
            
            # [Batch, 1, Embed_Dim]
            d = self.danger_projection(d)            
            x = x + d

        # The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        # attention mechanism, actual transformer
        out = self.transformer(x, mask=mask, is_causal=True)
        pred_note = self.fc_note(out)
        pred_dur = self.fc_dur(out)
        
        return pred_note, pred_dur