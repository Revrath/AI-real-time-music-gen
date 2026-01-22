import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

"""
AI is guessing the next note of a song based on previous SEQUENCE_LENGTH notes. 
It returns probabiblity distribution over all possible notes.
Input is
 - Normalized note value (0 to 1), based on all possible notes in dataset
 - Danger level (0 to 1)

"""

BATCH_SIZE = 128
EPOCHS = 50 
SEQUENCE_LENGTH = 50 # Must match preprocess.py
def train_network():
    with open('music_data.pkl', 'rb') as f:
        data = pickle.load(f)
        raw_notes = data['notes']
        raw_labels = data['labels']

    pitchnames = sorted(set(raw_notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    n_vocab = len(pitchnames)
    network_input = []
    network_output = []

    # CREATE WINDOWS
    for i in range(0, len(raw_notes) - SEQUENCE_LENGTH):
        sequence_in = raw_notes[i:i + SEQUENCE_LENGTH]
        sequence_out = raw_notes[i + SEQUENCE_LENGTH]
        danger_level = raw_labels[i + SEQUENCE_LENGTH] 
        
        # Convert notes to integers
        input_seq = [note_to_int[char] for char in sequence_in]
        
        complex_input = []
        for note_val in input_seq:
            # Normalize note (0-1) and append danger level (0-1)
            norm_note = note_val / len(pitchnames)
            complex_input.append([norm_note, danger_level])
            
        network_input.append(complex_input)
        network_output.append(note_to_int[sequence_out])

    X = np.array(network_input) 
    y = to_categorical(network_output, num_classes=n_vocab)

    print(X.shape) 
    # 
    print(y.shape) 

    model = create_model(X, n_vocab)

    filepath = "weights-{epoch:02d}-{loss:.4f}-seq_len-"+f"{SEQUENCE_LENGTH}"+".keras"
    checkpoint = ModelCheckpoint(
        filepath, 
        monitor='loss',
        verbose = 1,
        save_best_only=True,
        mode = 'min'
    )
    callbacks_list = [checkpoint]

    print("Starting training... (Press Ctrl+C to stop safely)")
    model.fit(
            X, y, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            callbacks=callbacks_list
        )

def create_model(inputs, n_vocab):
    model = Sequential()
    
    model.add(LSTM(
        128,
        input_shape = (inputs.shape[1], inputs.shape[2]),
        return_sequences = True
    ))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    model.summary()
    return model

if __name__ == '__main__':
    train_network()