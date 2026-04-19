"""
inference/predict.py
Run stair detection inference on a single point cloud.
"""

import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.stair_cnn import StairCNN
from preprocessing.voxelizer import Voxelizer

CHECKPOINT = "checkpoints/best_model.pth"
LABELS = ["flat_ground", "stairs"]

class StairDetector:
    def __init__(self, checkpoint: str = CHECKPOINT):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.voxelizer = Voxelizer()
        self.model = StairCNN().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint,
                                              map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {checkpoint} on {self.device}")

    def predict(self, points: np.ndarray) -> dict:
        """
        Run inference on a single point cloud.
        points: Nx3 or Nx4 numpy array
        Returns dict with label and confidence.
        """
        voxel = self.voxelizer.to_voxel_grid(points)
        tensor = torch.tensor(voxel).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())

        return {
            "label":       LABELS[pred],
            "confidence":  float(probs[pred]),
            "stairs_prob": float(probs[1])
        }
