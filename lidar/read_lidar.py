"""
Subscribe to Unitree Go2 L1 LiDAR point cloud via DDS.
Usage: python read_lidar.py <network_interface>
Example: python read_lidar.py eth0
"""

import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

TOPIC_CLOUD = "rt/utlidar/cloud"

def parse_point_cloud(msg: PointCloud2_) -> np.ndarray:
    """Parse raw PointCloud2 message into Nx4 numpy array (x, y, z, intensity)."""
    raw = bytes(msg.data)          # msg.data is sequence[uint8]
    point_step = msg.point_step    # bytes per point (typically 16)
    num_points = msg.width         # msg.height is 1 for unordered clouds

    dt = np.dtype([
        ('x',         np.float32),
        ('y',         np.float32),
        ('z',         np.float32),
        ('intensity', np.float32),
    ])
    points = np.frombuffer(raw, dtype=dt)
    return points

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)  # auto-detect interface

    sub = ChannelSubscriber(TOPIC_CLOUD, PointCloud2_)
    sub.Init()

    print(f"Subscribed to: {TOPIC_CLOUD}")
    print("Waiting for point clouds... (Ctrl+C to stop)\n")

    try:
        while True:
            msg = sub.Read()
            if msg is not None:
                points = parse_point_cloud(msg)
                xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)

                print(f"Received point cloud:")
                print(f"  timestamp : {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
                print(f"  frame     : {msg.header.frame_id}")
                print(f"  width     : {msg.width}")
                print(f"  height    : {msg.height}")
                print(f"  point_step: {msg.point_step} bytes")
                print(f"  points    : {len(points)}")
                print(f"  x range   : [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]")
                print(f"  y range   : [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]")
                print(f"  z range   : [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
                print()
            else:
                time.sleep(0.01)  # small sleep to avoid busy-waiting
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        sub.Close()
