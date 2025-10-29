import os
import time
import json
import logging
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import det_curve

from data_feeder import ASVDataSet, load_data
from model import SSL_AASIST_Model

from config import ASV19, ASV5


# ---------------------- Utility Functions ---------------------- #

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_eer(y_true: list, y_score: list) -> float:
    """Compute Equal Error Rate (EER)."""
    fpr, fnr, _ = det_curve(y_true, y_score)
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2


def setup_logging(output_dir: str) -> None:
    """Configure logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training_log.log'),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def save_config(args: argparse.Namespace, output_dir: str) -> None:
    """Save training configuration to JSON."""
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Configuration saved to {config_path}")


# ---------------------- Training Functions ---------------------- #

def train(model: nn.Module, device: torch.device, loader: DataLoader, optimizer: optim.Optimizer) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss, total_samples = 0, 0
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    return total_loss / total_samples


def validate(model: nn.Module, device: torch.device, loader: DataLoader) -> Tuple[float, float]:
    """Evaluate the model and compute loss and EER."""
    model.eval()
    total_loss, total_samples = 0, 0
    all_scores, all_labels = [], []
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            scores = output[:, 1].cpu().numpy()
            labels = target.cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels)

    avg_loss = total_loss / total_samples
    eer = get_eer(all_labels, all_scores)
    return avg_loss, eer


# ---------------------- Main Function ---------------------- #

def main():
    parser = argparse.ArgumentParser(description='Mamba Trials Setup')
    parser.add_argument("-o", "--out_fold", type=str, default='./', help="Output folder")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup', type=float, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default="ASV19")
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--kernel_sizes', type=str, default="3,5,7,15")
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()
    args.out_fold = f"models/ConvAdapterMultiHead/{args.kernel_sizes}/{args.seed}_{args.dataset}_{args.kernel_sizes}"

    # Prepare output directories and logging
    os.makedirs(os.path.join(args.out_fold, 'checkpoint'), exist_ok=True)
    setup_logging(args.out_fold)
    save_config(args, args.out_fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Dataset paths
    if args.dataset == "ASV19":
        train_flac = ASV19["train_flac"]
        dev_flac = ASV19["dev_flac"]
        train_protocol = ASV19["train_protocol"]
        dev_protocol = ASV19["dev_protocol"]

    elif args.dataset == "ASV5":  # ASV5
        train_flac = ASV5["train_flac"]
        dev_flac = ASV5["dev_flac"]
        train_protocol = ASV5["train_protocol"]
        dev_protocol = ASV5["dev_protocol"]

    # Load model
    model = SSL_AASIST_Model(vars(args)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    # Load datasets
    _, train_data, train_label = load_data(train_flac, "train", train_protocol, mode="train")
    train_dataset = ASVDataSet(train_data, train_label, mode="train")
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              num_workers=1, 
                              pin_memory=True
                              )

    _, dev_data, dev_label = load_data(dev_flac, "dev", dev_protocol, mode="train")
    dev_dataset = ASVDataSet(dev_data, dev_label, mode="dev")
    dev_loader = DataLoader(dev_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=1, 
                            pin_memory=True
                            )

    # Training loop with early stopping
    patience, counter, best_loss = args.early_stopping, 0, float('inf')
    writer = open(f"{args.out_fold}/writer.txt", "w")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs}")

        train_loss = train(model, device, train_loader, optimizer)
        val_loss, eer = validate(model, device, dev_loader)

        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val EER: {eer*100:.2f}%")
        writer.write(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val EER {eer*100:.2f}%\n")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            # Save checkpoints
            torch.save(model.state_dict(), os.path.join(args.out_fold, 'checkpoint', f'senet_epoch_{epoch}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.out_fold, 'checkpoint', f'op_epoch_{epoch}.pt'))
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.out_fold, 'SSL_best.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.out_fold, 'op.pt'))
                logging.info("Best model saved")
                writer.write("Best model saved\n")
        else:
            counter += 1
            logging.info(f"No improvement for {counter} epoch(s)")
            writer.write(f"No improvement for {counter} epoch(s)\n")

        if counter >= patience:
            logging.info(f"Early stopping triggered after {patience} epochs without improvement")
            writer.write(f"Early stopping triggered after {patience} epochs\n")
            break

        writer.flush()

    writer.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = round(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    print(f"Total time: {h}h:{m}m:{s}s")
