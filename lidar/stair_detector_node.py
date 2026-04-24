"""
lidar/stair_detector_node.py
Live stair detection with RViz2 visualization.
Uses SVM pre-filter + CNN for robust stair detection.
Publishes point cloud + red marker when stairs detected.
"""
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as UnitreeCloud

sys.path.append('/home/user/Desktop/stair_detection')
from inference.predict import StairDetector
from preprocessing.voxelizer import Voxelizer
from preprocessing.feature_extractor import extract_features
from models.stair_svm import StairSVM

TOPIC_CLOUD   = "rt/utlidar/cloud"
TOPIC_OUT     = "/utlidar/cloud"
TOPIC_MARKER  = "/stair_markers"
CHECKPOINT    = "/home/user/Desktop/stair_detection/checkpoints/best_model.pth"
SVM_CHECKPOINT = "/home/user/Desktop/stair_detection/checkpoints/stair_svm.pkl"

class StairDetectorNode(Node):
    def __init__(self):
        super().__init__('stair_detector')

        # Publishers
        self.cloud_pub  = self.create_publisher(PointCloud2, TOPIC_OUT, 10)
        self.marker_pub = self.create_publisher(MarkerArray, TOPIC_MARKER, 10)

        # CNN stair detector
        self.detector  = StairDetector(checkpoint=CHECKPOINT)
        self.voxelizer = Voxelizer()

        # SVM pre-filter
        self.svm = StairSVM()
        if os.path.exists(SVM_CHECKPOINT):
            self.svm.load(SVM_CHECKPOINT)
            self.svm_ready = True
            self.get_logger().info("SVM pre-filter loaded")
        else:
            self.svm_ready = False
            self.get_logger().warn(
                "SVM checkpoint not found — running CNN only. "
                "Train SVM first with: python training/train_svm.py"
            )

        # Detection state
        self.stairs_detected = False
        self.stairs_prob     = 0.0
        self.last_stair_xyz  = None
        self.stair_center_history = []
        self.history_size = 10

        self.get_logger().info("Stair detector node started")
        self.get_logger().info(f"Point cloud   → {TOPIC_OUT}")
        self.get_logger().info(f"Stair markers → {TOPIC_MARKER}")

    def process(self, msg: UnitreeCloud):
        # --- Publish point cloud ---
        ros_msg = PointCloud2()
        ros_msg.header = Header()
        ros_msg.header.stamp = self.get_clock().now().to_msg()
        ros_msg.header.frame_id = msg.header.frame_id
        ros_msg.height       = msg.height
        ros_msg.width        = msg.width
        ros_msg.point_step   = msg.point_step
        ros_msg.row_step     = msg.row_step
        ros_msg.is_dense     = msg.is_dense
        ros_msg.is_bigendian = msg.is_bigendian
        ros_msg.data = bytes(msg.data)
        for f in msg.fields:
            pf = PointField()
            pf.name     = f.name
            pf.offset   = f.offset
            pf.datatype = f.datatype
            pf.count    = f.count
            ros_msg.fields.append(pf)
        self.cloud_pub.publish(ros_msg)

        # --- Parse point cloud ---
        raw = bytes(msg.data)
        dt = np.dtype([('x', np.float32), ('y', np.float32),
                       ('z', np.float32), ('intensity', np.float32)])
        points = np.frombuffer(raw, dtype=dt)
        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 0] < 250.0)
        xyz = xyz[valid]

        if len(xyz) < 50:
            self.stairs_detected = False
            self.stairs_prob = 0.0
            self.publish_marker(xyz, ros_msg.header)
            return

        # --- Stage 1: SVM pre-filter ---
        if self.svm_ready:
            features = extract_features(xyz)
            if features is None:
                self.stairs_detected = False
                self.stairs_prob = 0.0
                self.publish_marker(xyz, ros_msg.header)
                return

            svm_result = self.svm.predict(features)
            svm_prob = svm_result['stairs_prob']

            if svm_prob < 0.4:
                # SVM confident this is NOT stairs — skip CNN
                self.stairs_detected = False
                self.stairs_prob = 0.0
                self.publish_marker(xyz, ros_msg.header)
                return

            # SVM thinks maybe stairs — pass to CNN
            self.get_logger().debug(
                f"SVM passed scan (stairs_prob={svm_prob:.2f}) → running CNN"
            )

        # # --- Stage 2: CNN classification ---
        # result = self.detector.predict(xyz)
        # self.stairs_prob     = result['stairs_prob']
        # self.stairs_detected = self.stairs_prob > 0.8

        # if self.stairs_detected:
        #     self.last_stair_xyz = xyz
        #     if self.svm_ready:
        #         self.get_logger().info(
        #             f"⚠️  STAIRS DETECTED — "
        #             f"SVM: {svm_prob:.0%} | "
        #             f"CNN: {result['confidence']:.0%}"
        #         )
        #     else:
        #         self.get_logger().info(
        #             f"⚠️  STAIRS DETECTED — "
        #             f"CNN confidence: {result['confidence']:.0%}"
        #         )
        # --- Stage 2: CNN classification ---
        result = self.detector.predict(xyz)
        self.stairs_prob = result['stairs_prob']

        # Require BOTH SVM and CNN to agree
        if self.svm_ready:
            self.stairs_detected = (svm_prob > 0.7) and (result['stairs_prob'] > 0.8)
        else:
            self.stairs_detected = result['stairs_prob'] > 0.8

        if self.stairs_detected:
            self.last_stair_xyz = xyz
            self.get_logger().info(
                f"⚠️  STAIRS DETECTED — "
                f"SVM: {svm_prob:.0%} | "
                f"CNN: {result['confidence']:.0%}"
            )
        self.get_logger().info(f"DEBUG: stairs_detected={self.stairs_detected} stairs_prob={self.stairs_prob:.2f}")
        # --- Publish marker ---
        self.publish_marker(xyz, ros_msg.header)

    def publish_marker(self, xyz: np.ndarray, header):
        markers = MarkerArray()

        if self.stairs_detected and len(xyz) > 0:
            # Crop to stair region
            # front = xyz[(xyz[:, 0] > 0.3) & (xyz[:, 0] < 3.0)]
            # if len(front) > 10:
            #     x_min = float(front[:, 0].min())
            #     x_max = float(front[:, 0].max())
            #     y_min = float(front[:, 1].min())
            #     y_max = float(front[:, 1].max())
            #     z_min = float(front[:, 2].min())
            #     z_max = float(front[:, 2].max())
            #     cx = (x_min + x_max) / 2
            #     cy = (y_min + y_max) / 2
            #     cz = (z_min + z_max) / 2
            front = xyz[(xyz[:, 0] > 0.3) & (xyz[:, 0] < 3.0)]
            if len(front) > 10:
                # Option 2 — crop to stair points only (above ground level)
                stair_points = front[front[:, 2] > front[:, 2].min() + 0.05]
                if len(stair_points) < 10:
                    stair_points = front  # fallback to all points

                x_min = float(stair_points[:, 0].min())
                x_max = float(stair_points[:, 0].max())
                y_min = float(stair_points[:, 1].min())
                y_max = float(stair_points[:, 1].max())
                z_min = float(stair_points[:, 2].min())
                z_max = float(stair_points[:, 2].max())

                # Option 1 — smooth center over last N detections
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                cz = (z_min + z_max) / 2

                self.stair_center_history.append([cx, cy, cz])
                if len(self.stair_center_history) > self.history_size:
                    self.stair_center_history.pop(0)
                cx, cy, cz = np.mean(self.stair_center_history, axis=0)

                # --- Red bounding box ---
                box = Marker()
                box.header  = header
                box.ns      = "stair_box"
                box.id      = 0
                box.type    = Marker.LINE_LIST
                box.action  = Marker.ADD
                box.scale.x = 0.03
                box.color   = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
                box.lifetime.sec = 1

                corners = [
                    [x_min, y_min, z_min], [x_max, y_min, z_min],
                    [x_max, y_max, z_min], [x_min, y_max, z_min],
                    [x_min, y_min, z_max], [x_max, y_min, z_max],
                    [x_max, y_max, z_max], [x_min, y_max, z_max],
                ]
                edges = [
                    (0,1),(1,2),(2,3),(3,0),
                    (4,5),(5,6),(6,7),(7,4),
                    (0,4),(1,5),(2,6),(3,7),
                ]
                for i, j in edges:
                    box.points.append(Point(
                        x=corners[i][0], y=corners[i][1], z=corners[i][2]))
                    box.points.append(Point(
                        x=corners[j][0], y=corners[j][1], z=corners[j][2]))
                markers.markers.append(box)

                # --- Red arrow pointing down to stairs ---
                arrow = Marker()
                arrow.header  = header
                arrow.ns      = "stair_arrow"
                arrow.id      = 1
                arrow.type    = Marker.ARROW
                arrow.action  = Marker.ADD
                arrow.scale.x = 0.1
                arrow.scale.y = 0.2
                arrow.scale.z = 0.3
                arrow.color   = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                arrow.lifetime.sec = 1
                arrow.points.append(Point(x=cx, y=cy, z=cz + 1.0))
                arrow.points.append(Point(x=cx, y=cy, z=cz + 0.1))
                markers.markers.append(arrow)

                # --- Text label ---
                text = Marker()
                text.header  = header
                text.ns      = "stair_text"
                text.id      = 2
                text.type    = Marker.TEXT_VIEW_FACING
                text.action  = Marker.ADD
                text.pose.position.x = cx
                text.pose.position.y = cy
                text.pose.position.z = cz + 1.3
                text.scale.z = 0.3
                text.color   = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                text.text    = f"STAIRS {self.stairs_prob:.0%}"
                text.lifetime.sec = 1
                markers.markers.append(text)

        else:
            self.stair_center_history = []  # reset history when no stairs
            # Clear all markers
            for ns, mid in [("stair_box", 0), ("stair_arrow", 1),
                            ("stair_text", 2)]:
                clear = Marker()
                clear.header = header
                clear.ns     = ns
                clear.id     = mid
                clear.action = Marker.DELETE
                markers.markers.append(clear)

        self.marker_pub.publish(markers)


import os

def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    rclpy.init()
    node = StairDetectorNode()

    sub = ChannelSubscriber(TOPIC_CLOUD, UnitreeCloud)
    sub.Init()

    print(f"\nSubscribed to: {TOPIC_CLOUD}")
    print("Publishing to RViz2...")
    print("Add these topics in RViz2:")
    print("  1. PointCloud2  → /utlidar/cloud")
    print("  2. MarkerArray  → /stair_markers")
    print("\nCtrl+C to stop.\n")

    try:
        while rclpy.ok():
            msg = sub.Read()
            if msg is not None:
                node.process(msg)
            else:
                time.sleep(0.01)
            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        sub.Close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()