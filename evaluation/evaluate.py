import time
import os
import json
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from compute_eer import get_eer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_feeder import ASVDataSet, load_data
from model import SSL_AASIST_Model
from config import DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SHUFFLE, DEFAULT_OUT_DIR, TEST, BEST_CKPT, BEST_CONFIG


def save_evaluation(model, output_dir, test_loader, ids, device):
    model.eval()
    score_list, labels = [], []
    out_file = os.path.join(output_dir, "scores.txt")

    with torch.no_grad(), open(out_file, "w") as w:
        for data, label in tqdm(test_loader, desc="Evaluating data ..."):
            data, label = data.to(device), label.to(device)

            result = (model(data).to("cpu")[:, 1]).data.cpu().numpy().ravel() 
            score_list.extend(result)

            labels.extend(label.view(-1).type(torch.int64).to("cpu").tolist())

        for i, wav_id in enumerate(ids):
            lab = "spoof" if labels[i] == 0 else "bonafide"
            w.write(f"{wav_id} {lab} {score_list[i]}\n")

    return score_list, labels


def special_eer(scores, labels):
    return get_eer(labels, scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("-o", "--out_fold", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature_type", type=str, default="fft")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Load config.json (contains model setup)
    with open(BEST_CONFIG, "r") as f:
        config = json.load(f)

    # Load MLAAD data
    eval_ids, dev_data, dev_labels = load_data(
        flac_path=TEST["flac"],
        dataset="Eval",
        label_file=TEST["protocol"],
        ext=TEST["ext"]
    )

    dev_dataset = ASVDataSet(dev_data, dev_labels, mode="dev")
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False
    )

    # Choose correct model
    config["training"] = False
    model = SSL_AASIST_Model(config)

    model = model.to(DEVICE)

    checkpoint = torch.load(BEST_CKPT, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_keys = set(model.state_dict().keys())

    new_state_dict = {}

    for k, v in state_dict.items():
        # Check if the key exists in the model
        if k in model_keys:
            new_state_dict[k] = v
        elif f"aasist.{k}" in model_keys:  # try adding prefix
            new_state_dict[f"aasist.{k}"] = v

    # Load safely
    model.load_state_dict(new_state_dict, strict=True)


    # Evaluate
    scores, labels = save_evaluation(model, args.out_fold, dev_loader, eval_ids, DEVICE)
    EER = special_eer(scores, labels) * 100

    with open(os.path.join(args.out_fold, "results.txt"), "w") as w:
        w.write(f"EER results are {EER:.2f}\n")

    print(f"EER: {EER:.2f}")


if __name__ == "__main__":
    start = time.time()
    main()
    total = round(time.time() - start)
    print(f"Evaluation time: {total // 3600}:{(total % 3600) // 60}:{total % 60}")
