"""
lidar/stair_detector_node.py
Live stair detection with RViz2 visualization.
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

TOPIC_CLOUD  = "rt/utlidar/cloud"
TOPIC_OUT    = "/utlidar/cloud"
TOPIC_MARKER = "/stair_markers"
CHECKPOINT   = "/home/user/Desktop/stair_detection/checkpoints/best_model.pth"

class StairDetectorNode(Node):
    def __init__(self):
        super().__init__('stair_detector')

        # Publishers
        self.cloud_pub  = self.create_publisher(PointCloud2, TOPIC_OUT, 10)
        self.marker_pub = self.create_publisher(MarkerArray, TOPIC_MARKER, 10)

        # Stair detector
        self.detector  = StairDetector(checkpoint=CHECKPOINT)
        self.voxelizer = Voxelizer()

        # Detection state
        self.stairs_detected  = False
        self.stairs_prob      = 0.0
        self.last_stair_xyz   = None

        self.get_logger().info("Stair detector node started")
        self.get_logger().info(f"Point cloud  → {TOPIC_OUT}")
        self.get_logger().info(f"Stair markers → {TOPIC_MARKER}")

    def process(self, msg: UnitreeCloud):
        # --- Publish point cloud ---
        ros_msg = PointCloud2()
        ros_msg.header = Header()
        ros_msg.header.stamp = self.get_clock().now().to_msg()
        ros_msg.header.frame_id = msg.header.frame_id
        ros_msg.height     = msg.height
        ros_msg.width      = msg.width
        ros_msg.point_step = msg.point_step
        ros_msg.row_step   = msg.row_step
        ros_msg.is_dense   = msg.is_dense
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
            return

        # --- Run inference ---
        result = self.detector.predict(xyz)
        self.stairs_prob     = result['stairs_prob']
        self.stairs_detected = self.stairs_prob > 0.8

        if self.stairs_detected:
            self.last_stair_xyz = xyz
            self.get_logger().info(
                f"⚠️  STAIRS DETECTED — confidence: {result['confidence']:.1%}"
            )

        # --- Publish marker ---
        self.publish_marker(xyz, ros_msg.header)

    def publish_marker(self, xyz: np.ndarray, header):
        markers = MarkerArray()

        # --- Marker 1: Red bounding box around stair region ---
        if self.stairs_detected and len(xyz) > 0:
            # Crop to stair region (in front of robot)
            front = xyz[(xyz[:, 0] > 0.3) & (xyz[:, 0] < 3.0)]
            if len(front) > 10:
                x_min, x_max = float(front[:, 0].min()), float(front[:, 0].max())
                y_min, y_max = float(front[:, 1].min()), float(front[:, 1].max())
                z_min, z_max = float(front[:, 2].min()), float(front[:, 2].max())
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                cz = (z_min + z_max) / 2

                # Bounding box (LINE_LIST style)
                box = Marker()
                box.header = header
                box.ns     = "stair_box"
                box.id     = 0
                box.type   = Marker.LINE_LIST
                box.action = Marker.ADD
                box.scale.x = 0.03  # line width
                box.color   = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
                box.lifetime.sec = 1

                # 8 corners of bounding box
                corners = [
                    [x_min, y_min, z_min], [x_max, y_min, z_min],
                    [x_max, y_max, z_min], [x_min, y_max, z_min],
                    [x_min, y_min, z_max], [x_max, y_min, z_max],
                    [x_max, y_max, z_max], [x_min, y_max, z_max],
                ]
                edges = [
                    (0,1),(1,2),(2,3),(3,0),  # bottom
                    (4,5),(5,6),(6,7),(7,4),  # top
                    (0,4),(1,5),(2,6),(3,7),  # sides
                ]
                for i, j in edges:
                    box.points.append(Point(x=corners[i][0], y=corners[i][1], z=corners[i][2]))
                    box.points.append(Point(x=corners[j][0], y=corners[j][1], z=corners[j][2]))
                markers.markers.append(box)

                # Arrow pointing to stairs
                arrow = Marker()
                arrow.header = header
                arrow.ns     = "stair_arrow"
                arrow.id     = 1
                arrow.type   = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.scale.x = 0.1   # shaft diameter
                arrow.scale.y = 0.2   # head diameter
                arrow.scale.z = 0.3   # head length
                arrow.color   = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                arrow.lifetime.sec = 1
                # Arrow from above pointing down to stair center
                arrow.points.append(Point(x=cx, y=cy, z=cz + 1.0))  # start (above)
                arrow.points.append(Point(x=cx, y=cy, z=cz + 0.1))  # end (near stairs)
                markers.markers.append(arrow)

                # Text label
                text = Marker()
                text.header = header
                text.ns     = "stair_text"
                text.id     = 2
                text.type   = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = cx
                text.pose.position.y = cy
                text.pose.position.z = cz + 1.3
                text.scale.z = 0.3  # text height
                text.color   = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                text.text    = f"STAIRS {self.stairs_prob:.0%}"
                text.lifetime.sec = 1
                markers.markers.append(text)

        else:
            # Clear markers when no stairs
            clear = Marker()
            clear.header = header
            clear.ns     = "stair_box"
            clear.id     = 0
            clear.action = Marker.DELETE
            markers.markers.append(clear)

        self.marker_pub.publish(markers)


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
