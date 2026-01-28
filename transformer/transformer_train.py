import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_into_pytorch_tensors import MusicDataset
from transformer_model import MusicTransformer

DATA_PATH = "lstm-ai-music-gen/output/music_data_allff9_dangertreshold06.pkl"
# CPU friendly settings
SEQ_LEN = 50        
BATCH_SIZE = 64 # I need bigger batch size when having small step or it will see only one thing in one pass
EMBED_DIM = 256
NUM_HEADS = 4  
NUM_LAYERS = 3
EPOCHS = 40
LEARNING_RATE = 0.001
DROPOUT = 0.2

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset = MusicDataset(DATA_PATH, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    note_vocab_size = len(dataset.note_vocab)
    dur_vocab_size = len(dataset.dur_vocab)

    model = MusicTransformer(
        note_vocab_size=note_vocab_size,
        dur_vocab_size=dur_vocab_size, 
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
        dropout= DROPOUT 
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf') 
    
    model.train()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0
        
        # loader returns things from __getitem__ in MusicDataset
        for i, (x_note, x_dur, y_note, y_dur, label) in enumerate(loader):
            x_note, x_dur = x_note.to(device), x_dur.to(device)
            y_note, y_dur = y_note.to(device), y_dur.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred_note, pred_dur = model(x_note, x_dur, danger_level=label)
            loss_note = criterion(pred_note.view(-1, note_vocab_size), y_note.view(-1))
            loss_dur = criterion(pred_dur.view(-1, dur_vocab_size), y_dur.view(-1))
            loss = loss_note + loss_dur
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(loader)} | "
                    f"Total: {loss.item():.4f} | "
                    f"Note: {loss_note.item():.4f} | "
                    f"Dur: {loss_dur.item():.4f}")

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"lstm-ai-music-gen/output/best_model{avg_loss:.4f}.pth")
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

if __name__ == "__main__":
    train()