"""
preprocessing/feature_extractor.py
Extract geometric features from point clouds for SVM classification.
"""
import numpy as np
from scipy import stats

def extract_features(xyz: np.ndarray) -> np.ndarray:
    """
    Extract geometric features from a point cloud.
    Returns 1D feature vector.
    """
    # Crop to region of interest
    front = xyz[(xyz[:, 0] > 0.3) & (xyz[:, 0] < 3.0) & (xyz[:, 0] < 250.0)]
    
    if len(front) < 50:
        return None

    x, y, z = front[:, 0], front[:, 1], front[:, 2]

    # Z statistics
    z_mean    = np.mean(z)
    z_std     = np.std(z)
    z_range   = z.max() - z.min()
    z_skew    = float(stats.skew(z))
    z_kurt    = float(stats.kurtosis(z))

    # Correlation features
    corr_xz = np.corrcoef(x, z)[0, 1]
    corr_yz = np.corrcoef(y, z)[0, 1]

    # Z histogram features — stairs have distinct horizontal layers
    hist, _ = np.histogram(z, bins=10)
    hist_norm = hist / hist.sum()
    z_entropy = float(stats.entropy(hist_norm + 1e-10))
    populated_bins = int(np.sum(hist > len(front) * 0.05))

    # Point density and spread
    num_points = len(front)
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    # XZ plane features — stairs have stepped pattern
    # Divide X into 5 strips and compute mean Z per strip
    x_bins = np.linspace(x.min(), x.max(), 6)
    z_per_x_strip = []
    for i in range(5):
        mask = (x >= x_bins[i]) & (x < x_bins[i+1])
        if mask.sum() > 5:
            z_per_x_strip.append(np.mean(z[mask]))
    
    if len(z_per_x_strip) >= 2:
        z_strip_std  = float(np.std(z_per_x_strip))
        z_strip_range = float(np.max(z_per_x_strip) - np.min(z_per_x_strip))
        z_strip_mono = float(np.corrcoef(
            range(len(z_per_x_strip)), z_per_x_strip)[0, 1])
    else:
        z_strip_std = z_strip_range = z_strip_mono = 0.0

    features = np.array([
        z_mean, z_std, z_range, z_skew, z_kurt,
        corr_xz, corr_yz,
        z_entropy, populated_bins,
        num_points, x_range, y_range,
        z_strip_std, z_strip_range, z_strip_mono
    ], dtype=np.float32)

    return features
