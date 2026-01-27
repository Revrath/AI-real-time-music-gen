import pickle
import numpy as np
import torch
from music21 import instrument, note, stream, chord
import os
import random
from transformer_model import MusicTransformer

MODEL_PATH = "lstm-ai-music-gen\\output\\best_model1.6689.pth"
DATA_PATH = "lstm-ai-music-gen/output/music_data_danger_duration.pkl"
SAFETY_LEVEL = 0.0  # 0.0 = Battle, 1.0 = Safe
SEQUENCE_LENGTH = 32 
NOTES_TO_GENERATE = 100 

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2

# the higher the more chaotic/creative
TEMPERATURE_NOTE = 1.3
TEMPERATURE_DUR = 0.8

def sample_with_temperature(predictions, temperature):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

def load_resources():
    try:
        device = torch.device("cpu")
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        raw_notes = data['notes']
        raw_durations = data['durations']

        pitchnames = sorted(list(set(raw_notes)))
        note_to_int = {n: i for i, n in enumerate(pitchnames)}
        int_to_note = {i: n for i, n in enumerate(pitchnames)}
        
        dur_vocab = sorted(list(set(raw_durations)))
        dur_to_int = {d: i for i, d in enumerate(dur_vocab)}
        int_to_dur = {i: d for i, d in enumerate(dur_vocab)}

        model = MusicTransformer(
            note_vocab_size=len(pitchnames),
            dur_vocab_size=len(dur_vocab),
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            seq_len=SEQUENCE_LENGTH
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, raw_notes, raw_durations, int_to_note, note_to_int, int_to_dur, dur_to_int, device
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def generate_music():
    model, raw_notes, raw_durations, int_to_note, note_to_int, int_to_dur, dur_to_int, device = load_resources()

    start_idx = np.random.randint(0, len(raw_notes) - SEQUENCE_LENGTH - 1)

    # starting patterns
    pattern_note = [note_to_int[c] for c in raw_notes[start_idx : start_idx + SEQUENCE_LENGTH]]
    pattern_dur = [dur_to_int[d] for d in raw_durations[start_idx : start_idx + SEQUENCE_LENGTH]]

    print(f"Generating {NOTES_TO_GENERATE} notes with safety level {SAFETY_LEVEL}")
    prediction_output = []

    for i in range(NOTES_TO_GENERATE):
        input_note_tensor = torch.tensor([pattern_note], dtype=torch.long).to(device)
        input_dur_tensor = torch.tensor([pattern_dur], dtype=torch.long).to(device)
        current_danger = torch.tensor([SAFETY_LEVEL], dtype=torch.float).to(device)

        with torch.no_grad():
            pred_note_logits, pred_dur_logits = model(input_note_tensor, input_dur_tensor, danger_level=current_danger)

            last_note_logits = pred_note_logits[0, -1, :]
            last_dur_logits = pred_dur_logits[0, -1, :]
            
            prob_note = torch.softmax(last_note_logits, dim=0).cpu().numpy()
            prob_dur = torch.softmax(last_dur_logits, dim=0).cpu().numpy()
            
        idx_note = sample_with_temperature(prob_note, temperature=TEMPERATURE_NOTE)
        idx_dur = sample_with_temperature(prob_dur, temperature=TEMPERATURE_DUR)
        
        result_note = int_to_note[idx_note]
        result_dur = int_to_dur[idx_dur]
        
        prediction_output.append((result_note, result_dur))
        
        # Move sliding window
        pattern_note.append(idx_note)
        pattern_note = pattern_note[1:]
        
        pattern_dur.append(idx_dur)
        pattern_dur = pattern_dur[1:]
    
    save_to_midi(prediction_output)

def save_to_midi(tokens):
    offset = 0
    output_notes = []

    for token, duration_str in tokens:
        duration = float(duration_str)

        if '.' in token or token.isdigit():
            if '.' in token:
                notes_in_chord = [note.Note(int(n)) for n in token.split('.')]
                for n in notes_in_chord: 
                    n.storedInstrument = instrument.PipeOrgan()
                new_element = chord.Chord(notes_in_chord)
            else:
                new_element = note.Note(int(token))
                new_element.storedInstrument = instrument.PipeOrgan()
        else:
            new_element = note.Note(token) # fallback

        new_element.quarterLength = duration
        new_element.offset = offset
        output_notes.append(new_element)
        offset += duration

    output_filename = f"lstm-ai-music-gen/output/transformer_danger_{SAFETY_LEVEL}_number_{random.random():.4f}_tempNote_{TEMPERATURE_NOTE}_tempDur_{TEMPERATURE_DUR}.mid"
    if not os.path.exists('output'):
        os.makedirs('output')
    stream.Stream(output_notes).write('midi', fp=output_filename)
    print(f"Saved to '{output_filename}'")

if __name__ == '__main__':
    generate_music()