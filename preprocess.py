import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, pitch, stream
import os

SEQUENCE_LENGTH = 40        # window size (MATCH THE MODEL)
                            # its size determine remembering patterns - too small and chaos comes and galaxies burn, too big and model learns songs by heart
DANGER_THRESHOLD = 0.6
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

def get_global_density(midi_score):
    try:
        # flatten merges all parts 
        offsets = sorted(list(set(n.offset for n in midi_score.flatten().notes)))
        
        if len(offsets) < 2: return 1.0

        intervals = [offsets[i+1] - offsets[i] for i in range(len(offsets)-1)]
        
        return sum(intervals) / len(intervals)
        
    except:
        return 1.0 

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
        print("skipping, no reasonable track")
        return [], []
    else:
        notes_to_parse = target_part.flatten().notes

    song_notes = []
    song_durations = []
    # I need to sort by offsets for duration calculation (flatten might mess it up)
    sorted_elements = sorted(list(notes_to_parse), key=lambda x: x.offset)    
    rhythmic_intervals = []

    for i, element in enumerate(sorted_elements):
        dur = quantize_duration(element.quarterLength)

        pitch_val = None
        if isinstance(element, note.Note):
            pitch_val = str(element.pitch.midi)
            
        elif isinstance(element, chord.Chord):
            if len(element.pitches) > 0:
                # lets try without chords
                top_note = max([n.midi for n in element.pitches])
                pitch_val = str(top_note)

        if pitch_val is not None:
            song_notes.append(pitch_val)
            song_durations.append(dur)

    return song_notes, song_durations

def get_notes():
    all_notes = []
    all_labels = [] # danger (0.0 to 1.0)
    all_durations = []

    folders_config = [
        ("lstm-ai-music-gen/FinalFantasy9/*.mid", 0.0), # here i was trying without danger labels
        # ("lstm-ai-music-gen/calmer/*.mid", 1.0)
    ]

    for folder_path, label_value in folders_config:
        files = glob.glob(folder_path)
        print(f"\nProcessing folder: {folder_path} (Label: {label_value}) - Found {len(files)} files")

        for file in files:
            try:
                print(f"Processing: {os.path.basename(file)}")
                midi = converter.parse(file)
                midi = transpose_to_c_major(midi)

                avg_global_interval = get_global_density(midi)
                melody_track, melody_durations = get_single_track_notes(midi)

                if len(melody_track) < 20 or len(melody_track) != len(melody_durations):
                    print(f"   -> SKIP {os.path.basename(file)}: track not suitable")
                    continue

                # quick notes == danger
                if avg_global_interval < DANGER_THRESHOLD:
                    label = 0.0  # danger
                    cat_str = "BATTLE"
                else:
                    label = 1.0  # safe
                    cat_str = "SAFE"

                print(f"Processing: {os.path.basename(file)} | Avg interval: {avg_global_interval:.2f}s -> {cat_str}")
                
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