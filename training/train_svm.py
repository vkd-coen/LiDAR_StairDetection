"""
training/train_svm.py
Train SVM on geometric features extracted from rosbags.
"""
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.rosbag_loader import load_bag
from preprocessing.feature_extractor import extract_features
from models.stair_svm import StairSVM

STAIRS_DIR    = "/opt/stair_data/stairs"
FLAT_DIR      = "/opt/stair_data/flat_ground"
SAVE_PATH     = "checkpoints/stair_svm.pkl"

def build_feature_dataset(bag_dir, label, msgs_per_bag=100):
    X, y = [], []
    for bag_name in sorted(os.listdir(bag_dir)):
        bag_path = os.path.join(bag_dir, bag_name)
        if not os.path.isdir(bag_path):
            continue
        clouds = load_bag(bag_path, max_msgs=msgs_per_bag)
        for pts in clouds:
            features = extract_features(pts)
            if features is not None:
                X.append(features)
                y.append(label)
    return X, y

def train():
    print("Extracting features from stairs bags...")
    X_stairs, y_stairs = build_feature_dataset(STAIRS_DIR, label=1)
    print(f"  {len(X_stairs)} stair samples")

    print("Extracting features from flat ground bags...")
    X_flat, y_flat = build_feature_dataset(FLAT_DIR, label=0)
    print(f"  {len(X_flat)} flat ground samples")

    X = np.array(X_stairs + X_flat)
    y = np.array(y_stairs + y_flat)
    print(f"\nTotal: {len(y)} samples")

    svm = StairSVM()
    svm.train(X, y)

    os.makedirs("checkpoints", exist_ok=True)
    svm.save(SAVE_PATH)

if __name__ == "__main__":
    train()
