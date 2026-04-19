
## LiDAR Data Collection (requires Unitree Go2 connection)

Find your network interface when connected to the robot:
```bash
ip addr
```

**Terminal 1 — Test raw point cloud data**
```bash
source /opt/ros2_lidar_env/bin/activate
cd ~/Desktop/stair_detection/lidar
python read_lidar.py eth0
```

**Terminal 2 — Launch ROS 2 bridge**
```bash
source /opt/ros2_lidar_env/bin/activate
cd ~/Desktop/stair_detection/lidar
python lidar_ros2_publisher.py eth0
```

**Terminal 3 — Visualize in RViz2**
```bash
source /opt/ros/humble/setup.bash
rviz2
```
In RViz2: Add → By topic → /utlidar/cloud → PointCloud2
