"""
preprocessing/dummy_data.py
Generates harder synthetic point clouds for stairs and flat ground.
Includes rotations, occlusions, partial views, and varied geometry.
"""

import numpy as np

def random_rotation_z(points: np.ndarray, angle_deg: float = None) -> np.ndarray:
    """Rotate point cloud around Z axis by a random angle."""
    if angle_deg is None:
        angle_deg = np.random.uniform(-45, 45)
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a, 0],
                  [sin_a,  cos_a, 0],
                  [0,      0,     1]])
    return points @ R.T

def random_dropout(points: np.ndarray, dropout_rate: float = None) -> np.ndarray:
    """Randomly remove a fraction of points to simulate occlusion."""
    if dropout_rate is None:
        dropout_rate = np.random.uniform(0.1, 0.4)
    mask = np.random.rand(len(points)) > dropout_rate
    return points[mask]

def generate_stairs(num_steps: int = None,
                    step_width: float = None,
                    step_depth: float = None,
                    step_height: float = None,
                    noise_std: float = None,
                    points_per_step: int = 200,
                    augment: bool = True) -> np.ndarray:
    """
    Generate a synthetic stair point cloud with realistic variations.
    Returns Nx3 numpy array.
    """
    # Randomize stair geometry each time
    num_steps    = num_steps    or np.random.randint(3, 8)
    step_width   = step_width   or np.random.uniform(0.8, 1.5)
    step_depth   = step_depth   or np.random.uniform(0.25, 0.40)
    step_height  = step_height  or np.random.uniform(0.15, 0.22)
    noise_std    = noise_std    or np.random.uniform(0.005, 0.03)

    all_points = []
    for i in range(num_steps):
        # Horizontal tread
        x = np.random.uniform(i * step_depth, (i + 1) * step_depth, points_per_step)
        y = np.random.uniform(-step_width / 2, step_width / 2, points_per_step)
        z = np.full(points_per_step, i * step_height)
        all_points.append(np.stack([x, y, z], axis=-1))

        # Vertical riser
        n = points_per_step // 2
        x_r = np.full(n, (i + 1) * step_depth)
        y_r = np.random.uniform(-step_width / 2, step_width / 2, n)
        z_r = np.random.uniform(i * step_height, (i + 1) * step_height, n)
        all_points.append(np.stack([x_r, y_r, z_r], axis=-1))

    points = np.vstack(all_points)
    points += np.random.normal(0, noise_std, points.shape)

    if augment:
        points = random_rotation_z(points)          # random orientation
        points = random_dropout(points)              # simulate occlusion
        # Random lateral offset (stairs not always centered)
        points[:, 1] += np.random.uniform(-0.3, 0.3)
        # Random distance offset (robot at different distances)
        points[:, 0] += np.random.uniform(0.0, 0.5)

    return points.astype(np.float32)

def generate_flat_ground(x_range: tuple = (0.3, 3.0),
                         y_range: tuple = (-1.5, 1.5),
                         noise_std: float = None,
                         num_points: int = None,
                         augment: bool = True) -> np.ndarray:
    """
    Generate realistic flat ground with variations.
    Includes slight slopes, bumps, and sparse regions.
    Returns Nx3 numpy array.
    """
    noise_std  = noise_std  or np.random.uniform(0.005, 0.025)
    num_points = num_points or np.random.randint(800, 2000)

    x = np.random.uniform(*x_range, num_points)
    y = np.random.uniform(*y_range, num_points)

    # Slight random slope to simulate uneven ground
    slope_x = np.random.uniform(-0.05, 0.05)
    slope_y = np.random.uniform(-0.05, 0.05)
    z = x * slope_x + y * slope_y

    points = np.stack([x, y, z], axis=-1)
    points += np.random.normal(0, noise_std, points.shape)

    if augment:
        points = random_dropout(points, dropout_rate=np.random.uniform(0.05, 0.25))
        # Occasionally add a small bump (not stairs but confusing)
        if np.random.rand() < 0.3:
            bump_x = np.random.uniform(0.5, 2.0)
            bump_mask = np.abs(points[:, 0] - bump_x) < 0.2
            points[bump_mask, 2] += np.random.uniform(0.02, 0.08)

    return points.astype(np.float32)

def generate_dataset(num_samples: int = 1000):
    """
    Generate harder balanced dataset of stairs and flat ground.
    Returns X (N, 1, 32, 32, 32) and y (N,) arrays.
    """
    from preprocessing.voxelizer import Voxelizer
    voxelizer = Voxelizer()

    X, y = [], []
    for i in range(num_samples // 2):
        # Stairs — label 1
        pts = generate_stairs(augment=True)
        voxel = voxelizer.to_voxel_grid(pts)
        X.append(voxel)
        y.append(1)

        # Flat ground — label 0
        pts = generate_flat_ground(augment=True)
        voxel = voxelizer.to_voxel_grid(pts)
        X.append(voxel)
        y.append(0)

        if (i + 1) % 100 == 0:
            print(f"  Generated {(i+1)*2}/{num_samples} samples...")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# """
# preprocessing/dummy_data.py
# Generates synthetic point clouds for stairs and flat ground.
# Used for pipeline testing before real LiDAR data is available.
# """

# import numpy as np

# def generate_stairs(num_steps: int = 5,
#                     step_width: float = 1.0,
#                     step_depth: float = 0.3,
#                     step_height: float = 0.18,
#                     noise_std: float = 0.01,
#                     points_per_step: int = 200) -> np.ndarray:
#     """
#     Generate a synthetic stair point cloud.
#     Returns Nx3 numpy array.
#     """
#     all_points = []
#     for i in range(num_steps):
#         # horizontal surface of each step
#         x = np.random.uniform(i * step_depth, (i + 1) * step_depth, points_per_step)
#         y = np.random.uniform(-step_width / 2, step_width / 2, points_per_step)
#         z = np.full(points_per_step, i * step_height)
#         step_points = np.stack([x, y, z], axis=-1)
#         all_points.append(step_points)

#         # vertical riser of each step
#         x_riser = np.full(points_per_step // 2, (i + 1) * step_depth)
#         y_riser = np.random.uniform(-step_width / 2, step_width / 2, points_per_step // 2)
#         z_riser = np.random.uniform(i * step_height, (i + 1) * step_height, points_per_step // 2)
#         riser_points = np.stack([x_riser, y_riser, z_riser], axis=-1)
#         all_points.append(riser_points)

#     points = np.vstack(all_points)
#     points += np.random.normal(0, noise_std, points.shape)  # add sensor noise
#     return points.astype(np.float32)

# def generate_flat_ground(x_range: tuple = (0.3, 3.0),
#                          y_range: tuple = (-1.5, 1.5),
#                          z_height: float = 0.0,
#                          noise_std: float = 0.01,
#                          num_points: int = 1500) -> np.ndarray:
#     """
#     Generate a synthetic flat ground point cloud.
#     Returns Nx3 numpy array.
#     """
#     x = np.random.uniform(*x_range, num_points)
#     y = np.random.uniform(*y_range, num_points)
#     z = np.full(num_points, z_height)
#     points = np.stack([x, y, z], axis=-1)
#     points += np.random.normal(0, noise_std, points.shape)
#     return points.astype(np.float32)

# def generate_dataset(num_samples: int = 500):
#     """
#     Generate balanced dataset of stairs and flat ground.
#     Returns X (N, 1, 32, 32, 32) and y (N,) arrays.
#     """
#     from preprocessing.voxelizer import Voxelizer
#     voxelizer = Voxelizer()

#     X, y = [], []
#     for _ in range(num_samples // 2):
#         # stairs — label 1
#         pts = generate_stairs()
#         X.append(voxelizer.to_voxel_grid(pts))
#         y.append(1)

#         # flat ground — label 0
#         pts = generate_flat_ground()
#         X.append(voxelizer.to_voxel_grid(pts))
#         y.append(0)

#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
