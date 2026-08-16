"""
preprocessing.py
Helper module for New Zealand Bird Call audio filtering and spectrogram generation.
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

# Pads shorter audio with zeros and trims longer audio to exactly 5 seconds
def pad_or_trim_audio(y, sr, target_duration = 3.0):
    target_length = int(sr * target_duration)

    if len(y) > target_length:
        return y[:target_length]
        
    elif len(y) < target_length:
        padding = target_length - len(y)
        return np.pad(y, (0, padding), mode='constant')
    return y


#This function removes low-frequency environmental noise (like wind, rain, and river rumbles) below 800 Hz
def apply_highpass_filter(y, sr, cutoff = 800.0, order = 5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype = 'high', analog = False, output = 'sos')
    return sosfilt(sos, y)


# Scans the audio and extracts only the non-silent, active sounds.
def extract_vocalisations(y, top_db = 20):
    # Find intervals where the audio is louder than the threshold
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # If the whole file is entirely too quiet, return original to avoid crashing
    if len(intervals) == 0:
        return y

    # Stitch the active, non-silent chunks back together
    salient_audio = np.concatenate([y[start:end] for start, end in intervals])
    return salient_audio

    

# Loads an audio file, applies a high-pass filter, extracts vocalizations and standardizes its length
def load_and_preprocess_audio(filepath, sr = 32000, target_duration = 3.0, cutoff = 800.0):
    try:
        # Load audio at native sample rate (32000 Hz based on the audit)
        y, _ = librosa.load(filepath, sr=sr)

        # Apply high-pass filter
        y_clean = apply_highpass_filter(y, sr, cutoff = cutoff)

        # Strip silence and grab the actual bird calls
        y_vocal = extract_vocalisations(y_clean, top_db=20)

        # Standardize length to 3 seconds
        y_final = pad_or_trim_audio(y_vocal, sr, target_duration = target_duration)

        return y_final, sr
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None


#Converts audio waveform into a Mel-Spectrogram
def extract_mel_spectrogram(y, sr, n_mels = 128, fmax = 8000):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)

    # Convert amplitude to logarithmic scale
    S_db = librosa.power_to_db(S, ref = np.max)

    return S_db