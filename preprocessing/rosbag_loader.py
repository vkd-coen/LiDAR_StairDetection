"""
preprocessing/rosbag_loader.py
Load labeled point clouds from recorded ROS 2 rosbags.
"""
import numpy as np
import sqlite3
import os
import struct
from preprocessing.voxelizer import Voxelizer

def parse_pointcloud2_bytes(raw: bytes) -> np.ndarray:
    """Parse raw CDR-serialized PointCloud2 bytes into Nx3 numpy array."""
    try:
        # Skip 4-byte CDR encapsulation header
        data = raw[4:]

        # Skip ROS2 PointCloud2 message header fields
        # header.stamp.sec (4) + header.stamp.nanosec (4) = 8 bytes
        # header.frame_id: 4 bytes length + N bytes string + padding
        offset = 8
        frame_id_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4 + frame_id_len
        # Align to 4 bytes
        if offset % 4 != 0:
            offset += 4 - (offset % 4)

        # height (4) + width (4)
        height = struct.unpack_from('<I', data, offset)[0]
        width  = struct.unpack_from('<I', data, offset + 4)[0]
        offset += 8

        # fields array: 4 bytes count + N field structs
        num_fields = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        for _ in range(num_fields):
            name_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4 + name_len
            if offset % 4 != 0:
                offset += 4 - (offset % 4)
            offset += 4  # uint32 offset_in_msg
            offset += 1  # uint8 datatype
            if offset % 4 != 0:
                offset += 4 - (offset % 4)  # CDR alignment before uint32 count
            offset += 4  # uint32 count

        # is_bigendian(1) + padding(3) + point_step(4) + row_step(4)
        offset += 1
        if offset % 4 != 0:
            offset += 4 - (offset % 4)
        point_step = struct.unpack_from('<I', data, offset)[0]
        offset += 8  # point_step + row_step

        # data array: 4 bytes length + point data
        data_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        raw_points = data[offset:offset + data_len]

        # Parse x,y,z from the first 12 bytes of each point (float32)
        num_points = data_len // point_step
        raw_arr = np.frombuffer(raw_points, dtype=np.uint8).reshape(num_points, point_step)
        xyz = np.frombuffer(raw_arr[:, :12].tobytes(), dtype=np.float32).reshape(num_points, 3)

        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 0] < 250.0)
        return xyz[valid].astype(np.float32)

    except Exception as e:
        print(f"  parse error: {e}")
        return None

def load_bag(bag_path: str, max_msgs: int = 200) -> list:
    """Load point clouds from a single rosbag."""
    db_files = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
    if not db_files:
        return []

    db_path = os.path.join(bag_path, db_files[0])
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT m.data FROM messages m
            JOIN topics t ON m.topic_id = t.id
            WHERE t.name = '/utlidar/cloud'
            LIMIT ?
        """, (max_msgs,))
        rows = cursor.fetchall()
    except Exception as e:
        print(f"  DB error in {bag_path}: {e}")
        return []
    finally:
        conn.close()

    clouds = []
    for row in rows:
        pts = parse_pointcloud2_bytes(bytes(row[0]))
        if pts is not None and len(pts) > 50:
            clouds.append(pts)

    return clouds

def generate_dataset_from_bags(stairs_bag_dir: str,
                                flat_bag_dir: str,
                                max_samples: int = 2000,
                                msgs_per_bag: int = 50):
    """
    Build training dataset from recorded rosbags.
    Returns X (N, 1, 32, 32, 32) and y (N,) arrays.
    """
    voxelizer = Voxelizer()
    X, y = [], []

    # Load stair samples
    print("Loading stair point clouds...")
    stair_bags = sorted(os.listdir(stairs_bag_dir))
    for bag_name in stair_bags:
        bag_path = os.path.join(stairs_bag_dir, bag_name)
        if not os.path.isdir(bag_path):
            continue
        clouds = load_bag(bag_path, max_msgs=msgs_per_bag)
        for pts in clouds:
            voxel = voxelizer.to_voxel_grid(pts)
            X.append(voxel)
            y.append(1)
        if len([yy for yy in y if yy == 1]) >= max_samples // 2:
            break
    print(f"  Loaded {sum(1 for yy in y if yy == 1)} stair samples")

    # Load flat ground samples
    print("Loading flat ground point clouds...")
    flat_bags = sorted(os.listdir(flat_bag_dir))
    for bag_name in flat_bags:
        bag_path = os.path.join(flat_bag_dir, bag_name)
        if not os.path.isdir(bag_path):
            continue
        clouds = load_bag(bag_path, max_msgs=msgs_per_bag)
        for pts in clouds:
            voxel = voxelizer.to_voxel_grid(pts)
            X.append(voxel)
            y.append(0)
        if len([yy for yy in y if yy == 0]) >= max_samples // 2:
            break
    print(f"  Loaded {sum(1 for yy in y if yy == 0)} flat ground samples")

    print(f"\nTotal dataset: {len(y)} samples")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# """
# preprocessing/rosbag_loader.py
# Load labeled point clouds from recorded ROS 2 rosbags.
# Includes data augmentation for better generalization.
# """
# import numpy as np
# import sqlite3
# import os
# import struct
# from preprocessing.voxelizer import Voxelizer

# def augment_point_cloud(points: np.ndarray) -> np.ndarray:
#     """Apply random augmentations to a point cloud."""
#     if len(points) == 0:
#         return points

#     # 1. Random rotation around Z axis (-30 to +30 degrees)
#     angle = np.random.uniform(-30, 30) * np.pi / 180
#     cos_a, sin_a = np.cos(angle), np.sin(angle)
#     R = np.array([[cos_a, -sin_a, 0],
#                   [sin_a,  cos_a, 0],
#                   [0,      0,     1]])
#     points = (R @ points.T).T

#     # 2. Random jitter
#     points += np.random.normal(0, 0.01, points.shape)

#     # 3. Random dropout (remove 10-30% of points)
#     dropout = np.random.uniform(0.1, 0.3)
#     mask = np.random.rand(len(points)) > dropout
#     if mask.sum() > 10:
#         points = points[mask]

#     # 4. Random X/Y translation (±20cm)
#     points[:, 0] += np.random.uniform(-0.2, 0.2)
#     points[:, 1] += np.random.uniform(-0.2, 0.2)

#     # 5. Random mirror along Y axis (left-right flip)
#     if np.random.rand() > 0.5:
#         points[:, 1] = -points[:, 1]

#     return points.astype(np.float32)

# def parse_pointcloud2_bytes(raw: bytes) -> np.ndarray:
#     """Parse raw CDR-serialized PointCloud2 bytes into Nx3 numpy array."""
#     try:
#         data = raw[4:]
#         offset = 8

#         # frame_id
#         frame_id_len = struct.unpack_from('<I', data, offset)[0]
#         offset += 4 + frame_id_len
#         if offset % 4 != 0:
#             offset += 4 - (offset % 4)

#         # height + width
#         height = struct.unpack_from('<I', data, offset)[0]
#         width  = struct.unpack_from('<I', data, offset + 4)[0]
#         offset += 8

#         # fields — fixed at 12 bytes each (bug fix: was 9)
#         num_fields = struct.unpack_from('<I', data, offset)[0]
#         offset += 4
#         for _ in range(num_fields):
#             name_len = struct.unpack_from('<I', data, offset)[0]
#             offset += 4 + name_len
#             if offset % 4 != 0:
#                 offset += 4 - (offset % 4)
#             offset += 12  # field_offset(4) + datatype(1) + padding(3) + count(4)

#         # is_bigendian
#         offset += 1
#         if offset % 4 != 0:
#             offset += 4 - (offset % 4)

#         # point_step + row_step
#         point_step = struct.unpack_from('<I', data, offset)[0]
#         offset += 8

#         # data array
#         data_len = struct.unpack_from('<I', data, offset)[0]
#         offset += 4
#         raw_points = data[offset:offset + data_len]

#         # Vectorized parsing (performance fix)
#         dt = np.dtype([
#             ('x', np.float32),
#             ('y', np.float32),
#             ('z', np.float32),
#             ('intensity', np.float32),
#         ])
#         num_points = data_len // point_step
#         raw_array = np.frombuffer(raw_points, dtype=np.uint8)
#         raw_array = raw_array[:num_points * point_step].reshape(num_points, point_step)
#         xyz_bytes = raw_array[:, :12].copy()
#         points = np.frombuffer(xyz_bytes.tobytes(), dtype=np.float32).reshape(num_points, 3)

#         valid = np.isfinite(points).all(axis=1) & (points[:, 0] < 250.0)
#         return points[valid].astype(np.float32)

#     except Exception as e:
#         print(f"  parse error: {e}")
#         return None

# def load_bag(bag_path: str, max_msgs: int = 200) -> list:
#     """Load point clouds from a single rosbag."""
#     db_files = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
#     if not db_files:
#         return []

#     db_path = os.path.join(bag_path, db_files[0])
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     try:
#         cursor.execute("""
#             SELECT m.data FROM messages m
#             JOIN topics t ON m.topic_id = t.id
#             WHERE t.name = '/utlidar/cloud'
#             LIMIT ?
#         """, (max_msgs,))
#         rows = cursor.fetchall()
#     except Exception as e:
#         print(f"  DB error in {bag_path}: {e}")
#         return []
#     finally:
#         conn.close()

#     clouds = []
#     for row in rows:
#         pts = parse_pointcloud2_bytes(bytes(row[0]))
#         if pts is not None and len(pts) > 50:
#             clouds.append(pts)

#     return clouds

# def generate_dataset_from_bags(stairs_bag_dir: str,
#                                 flat_bag_dir: str,
#                                 max_samples: int = 2000,
#                                 msgs_per_bag: int = 50,
#                                 augment: bool = False,
#                                 augment_factor: int = 3):
#     """
#     Build training dataset from recorded rosbags with optional augmentation.
#     Returns X (N, 1, 32, 32, 32) and y (N,) arrays.
#     """
#     voxelizer = Voxelizer()
#     X, y = [], []

#     def process_clouds(clouds, label, max_count):
#         for pts in clouds:
#             if len([yy for yy in y if yy == label]) >= max_count:
#                 break
#             # Original sample
#             voxel = voxelizer.to_voxel_grid(pts)
#             X.append(voxel)
#             y.append(label)

#             # Augmented samples
#             if augment:
#                 for _ in range(augment_factor):
#                     if len([yy for yy in y if yy == label]) >= max_count:
#                         break
#                     aug_pts = augment_point_cloud(pts.copy())
#                     voxel = voxelizer.to_voxel_grid(aug_pts)
#                     X.append(voxel)
#                     y.append(label)

#     # Load stair samples
#     print("Loading stair point clouds...")
#     stair_bags = sorted(os.listdir(stairs_bag_dir))
#     for bag_name in stair_bags:
#         bag_path = os.path.join(stairs_bag_dir, bag_name)
#         if not os.path.isdir(bag_path):
#             continue
#         clouds = load_bag(bag_path, max_msgs=msgs_per_bag)
#         process_clouds(clouds, label=1, max_count=max_samples // 2)
#         if len([yy for yy in y if yy == 1]) >= max_samples // 2:
#             break
#     print(f"  Loaded {sum(1 for yy in y if yy == 1)} stair samples")

#     # Load flat ground samples
#     print("Loading flat ground point clouds...")
#     flat_bags = sorted(os.listdir(flat_bag_dir))
#     for bag_name in flat_bags:
#         bag_path = os.path.join(flat_bag_dir, bag_name)
#         if not os.path.isdir(bag_path):
#             continue
#         clouds = load_bag(bag_path, max_msgs=msgs_per_bag)
#         process_clouds(clouds, label=0, max_count=max_samples // 2)
#         if len([yy for yy in y if yy == 0]) >= max_samples // 2:
#             break
#     print(f"  Loaded {sum(1 for yy in y if yy == 0)} flat ground samples")

#     print(f"\nTotal dataset: {len(y)} samples "
#           f"({sum(1 for yy in y if yy==1)} stairs, "
#           f"{sum(1 for yy in y if yy==0)} flat ground)")
#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)