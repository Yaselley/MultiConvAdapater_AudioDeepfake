# paths.py
import os

# ---------------------- ASV2019 ---------------------- #
ASV19 = {
    "train_flac": "/ds/audio/LA_19/ASVspoof2019_LA_train/flac/",
    "dev_flac": "/ds/audio/LA_19/ASVspoof2019_LA_dev/flac/",
    "train_protocol": "/netscratch/yelkheir/datasets/asvsoof/train.txt",
    "dev_protocol": "/netscratch/yelkheir/datasets/asvsoof/dev.txt"
}

# ---------------------- ASV5 ---------------------- #
ASV5 = {
    "train_flac": "/ds-slt/audio/ASVSpoof2024/flac_T/",
    "dev_flac": "/ds-slt/audio/ASVSpoof2024/flac_D/",
    "train_protocol": "/netscratch/yelkheir/datasets/asv5/asvspoof5.train.txt",
    "dev_protocol": "/netscratch/yelkheir/datasets/asv5/asvspoof5.dev.{seed}.txt"
}

# ---------------------- ASV5 ---------------------- #
TEST = {
    "flac": "/ds/audio/LA_19/ASVspoof2019_LA_train/flac/",
    "protocol": "/netscratch/yelkheir/datasets/asvsoof/train.txt",
    "ext": "flac"
}

# ---------------------- Default settings ---------------------- #
DEFAULT_MAX_LEN = 64600
SAMPLE_RATE = 16000
W2V_XLSR_CONFIG = "/netscratch/yelkheir/DeepFense/MultiConvAdapter/protocols/w2v_config.json"
W2V_XLSR_CKPT = "/netscratch/yelkheir/ssl_models/xlsr2_300m.pt"

DEVICE = "cuda"
BATCH_SIZE = 128
NUM_WORKERS = 1
PIN_MEMORY = True
SHUFFLE = False

# ---------------------- Default Output Folder ---------------------- #
DEFAULT_OUT_DIR = "/netscratch/yelkheir/standard/outputs/"

# ---------------------- BEST EPOCH ---------------------- #
BEST_CKPT = "/netscratch/yelkheir/DeepFense/MultiConvAdapter/protocols/CKPT/ASV19/SSL_best.pt"
BEST_CONFIG = "/netscratch/yelkheir/DeepFense/MultiConvAdapter/protocols/CKPT/ASV19/config.json"
