import pickle
import numpy as np
import torch
from music21 import instrument, note, stream, chord
import os
import random
from transformer_model import MusicTransformer

MODEL_PATH = "lstm-ai-music-gen/output/best_model.pth"
DATA_PATH = "lstm-ai-music-gen/output/music_data.pkl"
SAFETY_LEVEL = 0.0 
SEQUENCE_LENGTH = 32 
NOTES_TO_GENERATE = 100 

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2

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
            raw_notes = data['notes'] if isinstance(data, dict) else data
            
        pitchnames = sorted(set(raw_notes))
        n_vocab = len(pitchnames)
        note_to_int = {n: i for i, n in enumerate(pitchnames)}
        int_to_note = {i: n for i, n in enumerate(pitchnames)}
        
        model = MusicTransformer(n_vocab, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQUENCE_LENGTH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        return model, raw_notes, n_vocab, int_to_note, note_to_int, device
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def generate_music():
    model, raw_notes, n_vocab, int_to_note, note_to_int, device = load_resources()

    start_idx = np.random.randint(0, len(raw_notes) - SEQUENCE_LENGTH - 1)
    pattern = [note_to_int[c] for c in raw_notes[start_idx : start_idx + SEQUENCE_LENGTH]]
    
    print(f"Generating {NOTES_TO_GENERATE} notes with safety level {SAFETY_LEVEL}")
    prediction_output = []

    for i in range(NOTES_TO_GENERATE):
        prediction_input = torch.tensor([pattern], dtype=torch.long).to(device)
        current_danger = torch.tensor([SAFETY_LEVEL], dtype=torch.float).to(device)

        with torch.no_grad():
            output = model(prediction_input, danger_level=current_danger)
            last_logits = output[0, -1, :]
            prediction = torch.softmax(last_logits, dim=0).cpu().numpy()
            
        index = sample_with_temperature(prediction, temperature= 0.8)
        result = int_to_note[index]
        
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]
    
    save_to_midi(prediction_output)

def save_to_midi(tokens):
    offset = 0
    output_notes = []

    for token in tokens:
        if '.' in token or token.isdigit():
            if '.' in token:
                notes_in_chord = [note.Note(int(n)) for n in token.split('.')]
                for n in notes_in_chord: 
                    n.storedInstrument = instrument.PipeOrgan()
                new_element = chord.Chord(notes_in_chord)
                pitch_val = new_element.pitches[0].midi
            else:
                new_element = note.Note(int(token))
                new_element.storedInstrument = instrument.PipeOrgan()
                pitch_val = new_element.pitch.midi
        else:
            new_element = note.Note(token) # fallback
            pitch_val = 60

        if pitch_val < 50:
            duration = random.choice([1.0, 1.5, 2.0])
        elif pitch_val > 70:
            duration = random.choice([0.25, 0.5])
        else:
            duration = 0.5
            
        new_element.quarterLength = duration
        new_element.offset = offset + random.uniform(0, 0.05)
        output_notes.append(new_element)
        offset += duration

    output_filename = f"output/transformer_music_danger_{SAFETY_LEVEL}_random_{random.random():.4f}.mid"
    if not os.path.exists('output'):
        os.makedirs('output')
    stream.Stream(output_notes).write('midi', fp=output_filename)
    print(f"Saved to '{output_filename}'")

if __name__ == '__main__':
    generate_music()