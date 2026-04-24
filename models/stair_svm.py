"""
models/stair_svm.py
SVM classifier for stair detection using geometric features.
"""
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

class StairSVM:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                       probability=True))
        ])
        self.trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train SVM on feature vectors."""
        self.model.fit(X, y)
        self.trained = True
        scores = cross_val_score(self.model, X, y, cv=5)
        print(f"SVM cross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    def predict(self, features: np.ndarray) -> dict:
        """Predict on a single feature vector."""
        if not self.trained:
            raise RuntimeError("SVM not trained yet")
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        pred = int(probs.argmax())
        return {
            'label': 'stairs' if pred == 1 else 'flat_ground',
            'stairs_prob': float(probs[1]),
            'confidence': float(probs[pred])
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"SVM saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.trained = True
        print(f"SVM loaded from {path}")
