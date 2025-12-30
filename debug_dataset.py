import glob
import random
import numpy as np
from music21 import converter, instrument, note, chord, pitch, stream
import os


'''
To co dostawałem z predict.py brzmiało jak muzyka, ale z losowymi wstawkami bardzo niskich nut
Co się okazało
Funkcja midi.flat jest problematyczna bo z jednego dźwięku + bas robi dźwięk a potem bas
Słychać to wyraźnie na prostszym utworze (np super mario bros theme)

Po wyciągnięciu pierwszej ścieżki i zignorowaniu basów mario brzmi jak mario 
(nie licząc rytmu równy 0.5s generic dźwiękiu pianina)
'''


OUTPUT_FILE = "debug_preview.mid" 

def transpose_to_c_major(score):
    try:
        parts = score.getElementsByClass(stream.Part)
        if not parts:
            return score
            
        k = score.analyze('key')
        
        if k.mode == "major":
            interval = pitch.Interval(k.tonic, pitch.Pitch('C'))
        else:
            interval = pitch.Interval(k.tonic, pitch.Pitch('A'))
            
        transposed_score = score.transpose(interval)
        return transposed_score
    except Exception as e:
        print(f"transpose exception: {e}")
        return score
def get_single_track_notes(midi_score):
    try:
        parts = instrument.partitionByInstrument(midi_score)
    except:
        parts = None

    target_part = None

    if parts:
        # search for the first non-percussion track with notes - likely the melody.
        for p in parts.parts:
            inst = p.getInstrument()
            if 'Percussion' in str(type(inst)) or 'Drums' in str(type(inst)):
                continue
            
            # if has little notes, its not main track
            if len(p.flatten().notes) > 10:
                target_part = p
                print(f"taking first reasonable part: {p.partName}")
                break
    
    if target_part is None:
        print("taking whole score with flatten")
        notes_to_parse = midi_score.flatten().notes
    else:
        notes_to_parse = target_part.flatten().notes

    song_notes = []
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            song_notes.append(str(element.pitch.midi)) 
        elif isinstance(element, chord.Chord):
            sorted_pitches = sorted([n.midi for n in element.pitches])
            song_notes.append('.'.join(str(n) for n in sorted_pitches))
    print (song_notes)     
    return song_notes

def preview_preprocessing():
    target_file = "BlackMageVillage.mid" 
    print(f"Wybrano plik: {target_file}")

    midi = converter.parse(target_file)
    midi = transpose_to_c_major(midi)
    extracted_tokens = get_single_track_notes(midi)

    # tokens to midi file
    output_notes = []
    offset = 0
    
    for token in extracted_tokens:
        # if chord
        if ('.' in token) or token.isdigit():
            notes_in_chord = token.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.quarterLength = 0.5 
            output_notes.append(new_chord)
            
        # if note
        else:
            new_note = note.Note(token)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.quarterLength = 0.5 
            output_notes.append(new_note)
        
        # static time between notes
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=OUTPUT_FILE)
    
if __name__ == "__main__":
    preview_preprocessing()