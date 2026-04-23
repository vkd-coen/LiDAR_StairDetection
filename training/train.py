"""
training/train.py
Train StairCNN on synthetic (or real) dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report
import sys
import os

# add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.stair_cnn import StairCNN
# from preprocessing.dummy_data import generate_dataset
from preprocessing.rosbag_loader import generate_dataset_from_bags

# --- Config ---
NUM_SAMPLES   = 500
BATCH_SIZE    = 16
EPOCHS        = 30
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.2
SAVE_PATH     = "checkpoints/best_model.pth"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Data ---
    print("Generating synthetic dataset...")
    # X, y = generate_dataset(NUM_SAMPLES)
    X, y = generate_dataset_from_bags(
        stairs_bag_dir="/opt/stair_data/stairs",
        flat_bag_dir="/opt/stair_data/flat_ground",
        max_samples=2000,
        msgs_per_bag=50
        # ,augment=True,
        # augment_factor=3
    )
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

    print(f"Train samples: {train_size} | Val samples: {val_size}\n")

    # --- Model ---
    model = StairCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0

    # --- Training loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        # --- Validation ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(X_batch)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        train_acc = train_correct / train_size
        val_acc   = val_correct / val_size

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss/train_size:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss/val_size:.4f} Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → Saved best model (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print("\nClassification Report (last epoch):")
    print(classification_report(all_labels, all_preds,
                                target_names=["flat_ground", "stairs"]))

if __name__ == "__main__":
    train()
