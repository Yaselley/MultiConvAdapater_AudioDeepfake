import os
import random
import numpy as np
import librosa
from typing import Optional
from config import DEFAULT_MAX_LEN, SAMPLE_RATE


# ---------------------- Audio Processing Functions ---------------------- #

def pad(x: np.ndarray, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """
    Pad or truncate a 1D audio signal to a fixed length.
    
    Args:
        x: Input audio signal (1D numpy array).
        max_len: Target length in samples.
    
    Returns:
        Padded or truncated audio signal.
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(np.ceil(max_len / x_len))
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """
    Randomly pad or crop audio to a fixed length for data augmentation.
    
    Args:
        x: Input audio signal (1D numpy array).
        max_len: Target length in samples.
    
    Returns:
        Padded or randomly cropped audio signal.
    """
    x_len = x.shape[0]
    if x_len > max_len:
        start = np.random.randint(0, x_len - max_len)
        return x[start:start + max_len]
    num_repeats = int(np.ceil(max_len / x_len))
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def extract_fft(wav_path: str,) -> np.ndarray:
    """
    Load an audio file and prepare it for training (with random padding/cropping).
    
    Args:
        wav_path: Path to the audio file.
    
    Returns:
        Audio signal as a numpy array.
    """
    y, _ = librosa.load(wav_path, sr=16000)
    y = pad_random(y)
    return y


def extract_fft_eval_dev(wav_path: str) -> np.ndarray:
    """
    Load an audio file for evaluation/dev (fixed padding).
    
    Args:
        wav_path: Path to the audio file.
    
    Returns:
        Padded audio signal as a numpy array.
    """
    y, _ = librosa.load(wav_path, sr=16000)
    y = pad(y)
    return y


# ---------------------- Main ---------------------- #

def main():
    print("Feature extraction module loaded successfully")


if __name__ == "__main__":
    main()
