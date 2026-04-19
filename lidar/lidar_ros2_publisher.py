"""
Bridge Unitree Go2 L1 LiDAR DDS topic to ROS 2 for RViz2 visualization.
Usage: python lidar_ros2_publisher.py <network_interface>
Example: python lidar_ros2_publisher.py eth0
"""

import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as UnitreeCloud

TOPIC_IN  = "rt/utlidar/cloud"   # DDS topic from Go2
TOPIC_OUT = "/utlidar/cloud"     # ROS 2 topic for RViz2

class LidarBridgeNode(Node):
    def __init__(self):
        super().__init__('lidar_bridge')
        self.publisher = self.create_publisher(PointCloud2, TOPIC_OUT, 10)
        self.get_logger().info(f"Publishing to ROS 2 topic: {TOPIC_OUT}")

    def publish(self, msg: UnitreeCloud):
        ros_msg = PointCloud2()

        # Header
        ros_msg.header = Header()
        ros_msg.header.stamp = self.get_clock().now().to_msg()
        ros_msg.header.frame_id = msg.header.frame_id

        # Dimensions — width = num points, height = 1 for unordered cloud
        ros_msg.height     = msg.height
        ros_msg.width      = msg.width
        ros_msg.point_step = msg.point_step
        ros_msg.row_step   = msg.row_step
        ros_msg.is_dense   = msg.is_dense
        ros_msg.is_bigendian = msg.is_bigendian

        # Raw point data — msg.data is sequence[uint8]
        ros_msg.data = bytes(msg.data)

        # Copy PointField descriptors (x, y, z, intensity offsets)
        for f in msg.fields:
            pf = PointField()
            pf.name     = f.name
            pf.offset   = f.offset
            pf.datatype = f.datatype
            pf.count    = f.count
            ros_msg.fields.append(pf)

        self.publisher.publish(ros_msg)
        self.get_logger().info(
            f"Published {msg.width} points | frame: {msg.header.frame_id}"
        )

def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)  # auto-detect interface

    rclpy.init()
    node = LidarBridgeNode()

    sub = ChannelSubscriber(TOPIC_IN, UnitreeCloud)
    sub.Init()

    print(f"Subscribed to DDS topic : {TOPIC_IN}")
    print(f"Publishing to ROS2 topic: {TOPIC_OUT}")
    print("Open RViz2 and add PointCloud2 display on /utlidar/cloud")
    print("Ctrl+C to stop.\n")

    try:
        while rclpy.ok():
            msg = sub.Read()
            if msg is not None:
                node.publish(msg)
            else:
                time.sleep(0.01)  # avoid busy-waiting
            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        sub.Close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
