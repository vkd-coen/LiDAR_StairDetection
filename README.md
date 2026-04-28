# LiDAR Stair Detection — Unitree Go2

## Setup

### Prerequisites
- Ubuntu 22.04 LTS
- ROS 2 Humble
- Python 3.10
- Virtual environment at `/opt/ros2_lidar_env`

### Every Session — Network Setup
```bash
sudo nmcli device set eno2 managed no
sudo ip addr add 192.168.123.100/24 dev eno2
sudo ip link set eno2 up
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2"/></Interfaces></General></Domain></CycloneDDS>'
```

---

## LiDAR Data Collection (requires Unitree Go2 connection)

Find your network interface when connected to the robot:
```bash
ip addr
```

**Terminal 1 — Test raw point cloud data**
```bash
source /opt/ros2_lidar_env/bin/activate
cd ~/Desktop/stair_detection
python lidar/read_lidar.py eno2
```

**Terminal 2 — Launch stair detector + RViz2 bridge**
```bash
source /opt/ros2_lidar_env/bin/activate
source /opt/ros/humble/setup.bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2"/></Interfaces></General></Domain></CycloneDDS>'
cd ~/Desktop/stair_detection
python lidar/stair_detector_node.py eno2
```

**Terminal 3 — Visualize in RViz2**
```bash
source /opt/ros/humble/setup.bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2"/></Interfaces></General></Domain></CycloneDDS>'
rviz2
```

In RViz2:
- Fixed Frame → `utlidar_lidar`
- Add → By topic → `/utlidar/cloud` → PointCloud2
- Add → By topic → `/stair_markers` → MarkerArray

---

## Record Training Data

**Stairs:**
```bash
source /opt/ros/humble/setup.bash
ros2 bag record /utlidar/cloud -o /opt/stair_data/stairs/run1
```

**Flat ground / non-stair environments:**
```bash
source /opt/ros/humble/setup.bash
ros2 bag record /utlidar/cloud -o /opt/stair_data/flat_ground/run1
```

---

## Training

**Train CNN:**
```bash
source /opt/ros2_lidar_env/bin/activate
cd ~/Desktop/stair_detection
python training/train.py
```

**Train SVM:**
```bash
source /opt/ros2_lidar_env/bin/activate
cd ~/Desktop/stair_detection
python training/train_svm.py
```

---

## Structured Testing

```bash
source /opt/ros2_lidar_env/bin/activate
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno2"/></Interfaces></General></Domain></CycloneDDS>'
cd ~/Desktop/stair_detection

# Non-stair environments
python testing/run_test.py eno2 --scans 20 --label "corridor_wall" --expected no_stairs
python testing/run_test.py eno2 --scans 20 --label "open_room" --expected no_stairs
python testing/run_test.py eno2 --scans 20 --label "facing_door" --expected no_stairs

# Stair environments
python testing/run_test.py eno2 --scans 20 --label "stairs_center_1m" --expected stairs
python testing/run_test.py eno2 --scans 20 --label "stairs_center_1.5m" --expected stairs
python testing/run_test.py eno2 --scans 20 --label "stairs_center_2m" --expected stairs
python testing/run_test.py eno2 --scans 20 --label "stairs_left_15deg" --expected stairs
python testing/run_test.py eno2 --scans 20 --label "stairs_right_15deg" --expected stairs
```

Results are saved automatically to `testing/results.csv`.

---

## Dataset

Raw rosbags are stored locally at `/opt/stair_data/` and are not included
in this repository due to size constraints (3.7GB total).

### Dataset Summary

| Class | Runs | Approx. Scans | Size |
|---|---|---|---|
| Stairs | 60 | ~5,939 | 1.9 GB |
| Flat ground | 60 | ~6,000 | 1.8 GB |
| **Total** | **120** | **~11,939** | **3.7 GB** |

### Recording Details
- Robot: Unitree Go2
- Sensor: Unitree L1 LiDAR
- Format: ROS 2 rosbag (SQLite3)
- Topic: `/utlidar/cloud`
- Rate: ~15 Hz
- Point step: 32 bytes
- Environments: stairs, corridors, open rooms, doorways

---

## Project Structure
stair_detection/
├── lidar/
│   ├── read_lidar.py              # Raw point cloud reader
│   ├── lidar_ros2_publisher.py    # ROS2 bridge for RViz2
│   └── stair_detector_node.py    # Live detection + RViz2 markers
├── preprocessing/
│   ├── voxelizer.py               # 32x32x32 voxel grid
│   ├── rosbag_loader.py           # Loads rosbags for training
│   └── feature_extractor.py      # 15 geometric features for SVM
├── models/
│   ├── stair_cnn.py               # 3D CNN architecture
│   └── stair_svm.py               # SVM classifier
├── training/
│   ├── train.py                   # CNN training
│   └── train_svm.py               # SVM training
├── inference/
│   └── predict.py                 # StairDetector inference class
├── testing/
│   ├── run_test.py                # Structured test protocol
│   └── results.csv                # Test results
└── checkpoints/
├── best_model.pth             # Trained CNN checkpoint
└── stair_svm.pkl              # Trained SVM checkpoint

---

## Results Summary

| Metric | Value |
|---|---|
| CNN validation accuracy | 100% |
| SVM cross-validation accuracy | 84.1% ± 4.8% |
| Stair detection recall | 100% |
| Non-stair specificity | 87.9% |
| Overall F1 score | 0.922 |
