"""
preprocessing/voxelizer.py
Converts raw point cloud into a 3D voxel grid for CNN input.
"""

import numpy as np
import open3d as o3d

class Voxelizer:
    def __init__(self,
                 voxel_size: float = 0.1,       # size of each voxel in meters
                 x_range: tuple = (0.3, 3.0),   # forward range (in front of robot)
                 y_range: tuple = (-1.5, 1.5),  # lateral range
                 z_range: tuple = (-0.5, 1.0),  # vertical range
                 grid_size: tuple = (32, 32, 32) # output grid dimensions (X, Y, Z)
                 ):
        self.voxel_size = voxel_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.grid_size = grid_size

    # def crop(self, points: np.ndarray) -> np.ndarray:
    #     """Keep only points within the region of interest."""
    #     mask = (
    #         (points[:, 0] >= self.x_range[0]) & (points[:, 0] <= self.x_range[1]) &
    #         (points[:, 1] >= self.y_range[0]) & (points[:, 1] <= self.y_range[1]) &
    #         (points[:, 2] >= self.z_range[0]) & (points[:, 2] <= self.z_range[1])
    #     )
    #     return points[mask]
    def crop(self, points: np.ndarray) -> np.ndarray:
        """Keep only valid points within the region of interest."""
        # Filter out invalid sentinel values (255.0 = out of range)
        valid = (
            (points[:, 0] < 250.0) &  # filter 255 sentinel
            (points[:, 1] < 250.0) &
            (points[:, 2] < 250.0) &
            np.isfinite(points).all(axis=1)
        )
        points = points[valid]
        
        # Then crop to region of interest
        mask = (
            (points[:, 0] >= self.x_range[0]) & (points[:, 0] <= self.x_range[1]) &
            (points[:, 1] >= self.y_range[0]) & (points[:, 1] <= self.y_range[1]) &
            (points[:, 2] >= self.z_range[0]) & (points[:, 2] <= self.z_range[1])
        )
        return points[mask]

    def to_voxel_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Convert Nx3 point cloud to binary voxel grid of shape grid_size.
        Returns numpy array of shape (1, X, Y, Z) ready for CNN input.
        """
        # Step 1 — crop to region of interest
        cropped = self.crop(points[:, :3])  # use only x, y, z

        if len(cropped) == 0:
            return np.zeros((1, *self.grid_size), dtype=np.float32)

        # Step 2 — normalize to [0, grid_size] range
        x_norm = (cropped[:, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        y_norm = (cropped[:, 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
        z_norm = (cropped[:, 2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])

        # Step 3 — convert to grid indices
        xi = np.clip((x_norm * self.grid_size[0]).astype(int), 0, self.grid_size[0] - 1)
        yi = np.clip((y_norm * self.grid_size[1]).astype(int), 0, self.grid_size[1] - 1)
        zi = np.clip((z_norm * self.grid_size[2]).astype(int), 0, self.grid_size[2] - 1)

        # Step 4 — fill voxel grid
        grid = np.zeros(self.grid_size, dtype=np.float32)
        grid[xi, yi, zi] = 1.0

        return grid[np.newaxis, ...]  # shape: (1, 32, 32, 32)

    def visualize(self, points: np.ndarray):
        """Visualize cropped point cloud using Open3D."""
        cropped = self.crop(points[:, :3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cropped)
        o3d.visualization.draw_geometries([pcd])
