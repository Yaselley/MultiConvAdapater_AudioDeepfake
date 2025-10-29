import os
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from feature_extract import extract_fft, extract_fft_eval_dev


# ---------------------- Data Loading Functions ---------------------- #

def load_label(label_file: str) -> Tuple[Dict[str, int], List[str]]:
    """
    Load labels from a protocol file.
    
    Returns:
        labels: Dict mapping wav_id to 0 (spoof) or 1 (bonafide)
        wav_lists: List of wav_ids
    """
    labels = {}
    wav_lists = []
    encode = {'spoof': 0, 'bonafide': 1}

    with open(label_file, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                wav_id = parts[1]
                wav_lists.append(wav_id)
                try:
                    labels[wav_id] = encode[parts[4]]
                except IndexError:
                    labels[wav_id] = encode[parts[5]]

    return labels, wav_lists


def load_train_data(
    flac_path: str, dataset: str, label_file: str, ext: str = "flac"
) -> Tuple[List[str], List[str], List[int]]:
    """
    Load training data paths and labels.
    """
    labels, wav_lists = load_label(label_file)
    final_data, final_label, ids = [], [], []

    for wav_id in tqdm(wav_lists, desc=f"Loading {dataset} data"):
        wav_path = os.path.join(flac_path, f"{wav_id}.{ext}")
        if os.path.exists(wav_path):
            ids.append(wav_id)
            final_data.append(wav_path)
            final_label.append(labels[wav_id])
        else:
            print(f"Cannot open {wav_path}")

    return ids, final_data, final_label


def load_eval_data(dataset: str, scp_file: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Load evaluation data and metadata.
    
    Returns:
        wav_lists: list of wav_ids
        folder_list: mapping of wav_id to folder
        flag: mapping of wav_id to A00 or other identifier
    """
    wav_lists, folder_list, flag = [], {}, {}
    with open(scp_file, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                wav_id = parts[1]
                wav_lists.append(wav_id)
                folder_list[wav_id] = parts[-1]
                flag[wav_id] = parts[-2] if parts[-2] != '-' else 'A00'

    return wav_lists, folder_list, flag


def load_data(
    flac_path: str, dataset: str, label_file: str, mode: str = "train", ext: str = "flac"
) -> Tuple:
    """
    Unified function to load train or eval data.
    """
    if mode != "eval":
        return load_train_data(flac_path, dataset, label_file, ext)
    else:
        return load_eval_data(dataset, label_file)


# ---------------------- Dataset Class ---------------------- #

class ASVDataSet(Dataset):
    """
    PyTorch Dataset for ASV spoof detection.
    """

    def __init__(
        self,
        data: List[str],
        label: List[int],
        wav_ids: Optional[List[str]] = None,
        mode: str = "train",
        transform: bool = True,
        feature_type: str = "fft",
        aug: Optional = None,
        lengths: Optional[List[int]] = None,
    ):
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.mode = mode
        self.transform = transform
        self.feature_type = feature_type
        self.aug = aug
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        wav_path = self.data[idx]
        label = self.label[idx]

        if self.mode == "train":
            features = extract_fft(wav_path)
        else:
            features = extract_fft_eval_dev(wav_path)

        return features, label


if __name__ == "__main__":
    print("ASV Dataset module loaded successfully")
