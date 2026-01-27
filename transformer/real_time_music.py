import threading
import queue
import time
import mido
import torch
import numpy as np
import pickle
from transformer_model import MusicTransformer

# MODEL_PATH = "lstm-ai-music-gen/output/best_model0.6463.pth" # FF9
# MODEL_PATH = "lstm-ai-music-gen\output\\best_model1.8159.pth" # castlevania and nintendo calm songs (small dataset)
MODEL_PATH = "lstm-ai-music-gen\\output\\best_model0.2466.pth" # calmer songs and 4 battle castlevaia themes
DATA_PATH = "lstm-ai-music-gen/output/music_dataff9_only0label.pkl"
BPM = 120
QPM = 60 / BPM 

GLOBAL_STATE = {
    'danger_level': 0.0, # 0.0 = Battle, 1.0 = Safe
    'running': True
}

# queue as buffer between generator and player (lower is quicker response, higher is more stable playback)
note_queue = queue.Queue(maxsize=10) 

def load_ai():
    device = torch.device("cpu")
    
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    raw_notes = data['notes']
    raw_durations = data['durations']
    
    note_vocab = sorted(list(set(raw_notes)))
    dur_vocab = sorted(list(set(raw_durations)))
    
    note_to_int = {n: i for i, n in enumerate(note_vocab)}
    dur_to_int = {d: i for i, d in enumerate(dur_vocab)}
    int_to_note = {i: n for i, n in enumerate(note_vocab)}
    int_to_dur = {i: d for i, d in enumerate(dur_vocab)}
    
    model = MusicTransformer(len(note_vocab), len(dur_vocab), 256, 4, 2, 50, 0.2) # parameters must match trained model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    return model, raw_notes, raw_durations, note_to_int, dur_to_int, int_to_note, int_to_dur, device

def generator_thread(model, raw_notes, raw_durs, n2i, d2i, i2n, i2d, device):
    print("-> Wątek AI wystartował.")
    
    # Seed
    start_idx = np.random.randint(0, len(raw_notes) - 33)
    pattern_note = [n2i[n] for n in raw_notes[start_idx : start_idx + 32]]
    pattern_dur = [d2i[d] for d in raw_durs[start_idx : start_idx + 32]]
    
    def robust_sample(logits, temperature):
        logits = np.asarray(logits).astype('float64')
        
        logits = logits / temperature
        
        logits -= np.max(logits)
        
        # Softmax
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        
        probs = probs / np.sum(probs)
        
        # fix probabilities for numpy
        probs = probs * (1.0 - 1e-8)
        
        return np.argmax(np.random.multinomial(1, probs))

    while GLOBAL_STATE['running']:
        if note_queue.full():
            time.sleep(0.1)
            continue
        
        input_note = torch.tensor([pattern_note], dtype=torch.long).to(device)
        input_dur = torch.tensor([pattern_dur], dtype=torch.long).to(device)
        danger = torch.tensor([GLOBAL_STATE['danger_level']], dtype=torch.float).to(device)
        
        with torch.no_grad():
            out_note, out_dur = model(input_note, input_dur, danger_level=danger)
            
            logits_n = out_note[0, -1, :].cpu().numpy()
            logits_d = out_dur[0, -1, :].cpu().numpy()
            
            idx_n = robust_sample(logits_n, temperature=1)
            idx_d = robust_sample(logits_d, temperature=1)

        # decode
        note_str = i2n[idx_n]
        dur_str = i2d[idx_d]
        dur_val = float(dur_str)
        
        note_queue.put((note_str, dur_val))
        
        # move the window
        pattern_note.pop(0)
        pattern_note.append(idx_n)
        pattern_dur.pop(0)
        pattern_dur.append(idx_d)

def player_thread():
    
    try:
        port = mido.open_output() 
    except Exception as e:
        print(f"Błąd MIDI: {e}")
        return

    print(f"-> Our very special guest is {port.name}")
    weather = "safe"
    if GLOBAL_STATE['danger_level'] == 0.0:
        weather = "dangerous"

    print(f"the weather outside is {weather}")

    while GLOBAL_STATE['running']:
        try:
            token, duration = note_queue.get(timeout=1)
            
            notes_to_play = []
            if '.' in token:
                notes_to_play = [int(n) for n in token.split('.')]
            elif token.isdigit():
                notes_to_play = [int(token)]
            # else: token is pause or error
            
            # NOTE ON
            for n in notes_to_play:
                port.send(mido.Message('note_on', note=n, velocity=100))
            
            time_to_sleep = duration * QPM
            time.sleep(time_to_sleep)
            
            # NOTE OFF
            for n in notes_to_play:
                port.send(mido.Message('note_off', note=n, velocity=100))
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Błąd odtwarzania: {e}")
            break

if __name__ == "__main__":
    resources = load_ai()
    
    t_gen = threading.Thread(target=generator_thread, args=resources)
    t_play = threading.Thread(target=player_thread)
    
    t_gen.start()
    t_play.start()
    
    print("\n" + "="*40)
    print("  TONIGTH'S DJ IS... AI MUSIC GENERATOR!")
    print("  Type 'safe', 'battle' or 'exit'")
    print("="*40 + "\n")
    
    time.sleep(0.3)
    try:
        while True:
            cmd = input("Command > ").strip().lower()
            
            if cmd == 'exit':
                GLOBAL_STATE['running'] = False
                break
            elif cmd == 'safe':
                GLOBAL_STATE['danger_level'] = 1.0
                print("--- (Safe) ---")
            elif cmd == 'battle':
                GLOBAL_STATE['danger_level'] = 0.0
                print("--- (Danger) ---")
            else:
                print("???")
                
    except KeyboardInterrupt:
        GLOBAL_STATE['running'] = False

    t_gen.join()
    t_play.join()
    print("THAT IS ALL LADIES AND GENTLEMEN, THANK YOU FOR TUNING IN.")