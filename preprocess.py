import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, pitch, stream
import os

SEQUENCE_LENGTH = 40        # window size (MATCH THE MODEL)
                            # its size determine remembering patterns - too small and chaos comes and galaxies burn, too big and model learns songs by heart
def transpose_to_c_major(score):
    """
    analyse the key of the song and change it to C Major (if major) / A Minor (if minor)
    it makes easier for the AI to spot patterns
    """
    try:
        key = score.analyze('key')
        
        if key.mode == "major":
            interval = pitch.Interval(key.tonic, pitch.Pitch('C'))
        else:
            interval = pitch.Interval(key.tonic, pitch.Pitch('A'))
            
        transposed_score = score.transpose(interval)
        return transposed_score
    except:
        return score

# Quantize duration so ai will treat similar durations as same
COMMON_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
def quantize_duration(dur):
    return min(COMMON_DURATIONS, key=lambda x: abs(x - dur))


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
        print("taking whole score with flatten due to lack of reasonable part")
        notes_to_parse = midi_score.flatten().notes
    else:
        notes_to_parse = target_part.flatten().notes

    song_notes = []
    song_durations = []
    for element in notes_to_parse:
        dur = quantize_duration(element.quarterLength)
        if isinstance(element, note.Note):
            song_notes.append(str(element.pitch.midi)) 
            song_durations.append(dur)

        elif isinstance(element, chord.Chord):
            sorted_pitches = sorted([n.midi for n in element.pitches])
            song_notes.append('.'.join(str(n) for n in sorted_pitches))
            song_durations.append(dur)

    return song_notes, song_durations

def get_notes():
    all_notes = []
    all_labels = [] # danger (0.0 to 1.0)
    all_durations = []

    # in git history you can find simple script to distinguish safe vs battle midi files, which worked meh
    folders_config = [
        ("lstm-ai-music-gen/FinalFantasy9/*.mid", 0.0), # now im trying without danger labels
        # ("lstm-ai-music-gen/calmer/*.mid", 1.0)
    ]

    for folder_path, label_value in folders_config:
        files = glob.glob(folder_path)[:50]
        print(f"\nProcessing folder: {folder_path} (Label: {label_value}) - Found {len(files)} files")

        for file in files:
            try:
                print(f"Processing: {os.path.basename(file)}")
                midi = converter.parse(file)
                midi = transpose_to_c_major(midi)
                
                melody_track, melody_durations = get_single_track_notes(midi)

                label = label_value

                if len(melody_track) != len(melody_durations):
                    print("   -> SKIP: notes and durations length mismatch")
                    continue

                # just to make sure
                str_durations = [str(d) for d in melody_durations]

                all_notes.extend(melody_track)
                all_durations.extend(str_durations)
                all_labels.extend([label] * len(melody_track))
                
            except Exception as e:
                print(f"error parsing {file}: {e}")

    if not os.path.exists('lstm-ai-music-gen/output'):
        os.makedirs('lstm-ai-music-gen/output')

    with open('lstm-ai-music-gen/output/music_data.pkl', 'wb') as filepath:
        pickle.dump({'notes': all_notes, 'durations': all_durations, 'labels': all_labels}, filepath)
    
    print(f"\nSaved {len(all_notes)} notes to 'output/music_data.pkl'")

if __name__ == '__main__':
    get_notes()