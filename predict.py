import pickle
from matplotlib.pylab import rand
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from music21 import instrument, note, stream, chord
import os 
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
MODEL_PATH = "weights-04-4.6735-seq_len-50.keras"
DATA_PATH = "output/music_data.pkl"
SAFETY_LEVEL = 0.0 # Adjust between 1.0 (safe, slow notes) to 0.0 (dangerous, fast notes)
SEQUENCE_LENGTH = 50 # must match train.py
NOTES_TO_GENERATE = 100 # how long the music will be 

def sample_with_temperature(predictions, temperature):
    """
    predictions: The output array from the model (128 probabilities)
    temperature: 
       < 1.0 (Less Random / More Repetitive)
       > 1.0 (More Random / More Chaotic)
    """
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

def load_resources():
    try:
        model = load_model(MODEL_PATH)
        with open(DATA_PATH, 'rb') as f:
            raw_notes = pickle.load(f)['notes']
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    pitchnames = sorted(set(raw_notes))
    n_vocab = len(pitchnames)
    note_to_int = {n: i for i, n in enumerate(pitchnames)}
    int_to_note = {i: n for i, n in enumerate(pitchnames)}
    
    return model, raw_notes, n_vocab, int_to_note, note_to_int

def generate_music():
    model, raw_notes, n_vocab, int_to_note, note_to_int = load_resources()

    # pick a random sequence from the input as a starting point for prediction
    # it is possible that end of one song and start of another song will be mixed
    # the sequence is not added to the output
    start_idx = np.random.randint(0, len(raw_notes) - SEQUENCE_LENGTH - 1)
    pattern = [note_to_int[c] for c in raw_notes[start_idx : start_idx + SEQUENCE_LENGTH]]
    
    print(f"Generating {NOTES_TO_GENERATE} notes with safety level {SAFETY_LEVEL}")
    prediction_output = []

    for i in range(NOTES_TO_GENERATE):
        prediction_input = [[n / float(n_vocab), SAFETY_LEVEL] for n in pattern]
        prediction_input = np.reshape(prediction_input, (1, SEQUENCE_LENGTH, 2))
        
        prediction = model.predict(prediction_input, verbose=0)
        index = sample_with_temperature(prediction[0], temperature=1.0)
        result = int_to_note[index]
        
        # add new note and slide the window
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]
    save_to_midi(prediction_output)


def save_to_midi(tokens):
    offset = 0
    output_notes = []

    for token in tokens:
        if '.' in token or token.isdigit():
            notes_in_chord = [note.Note(int(n)) for n in token.split('.')]
            for n in notes_in_chord: 
                n.storedInstrument = instrument.PipeOrgan()
            new_element = chord.Chord(notes_in_chord)
            pitch_val = new_element.pitches[0].midi
        else:
            new_element = note.Note(token)
            new_element.storedInstrument = instrument.PipeOrgan()
            pitch_val = new_element.pitch.midi
        
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

    output_filename = f"output/ai_music_danger_{SAFETY_LEVEL}_random_number_{random.random()}.mid"
    stream.Stream(output_notes).write('midi', fp=output_filename)
    print(f"Saved to '{output_filename}'")

if __name__ == '__main__':
    generate_music()