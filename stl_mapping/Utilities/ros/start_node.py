import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
import os
from Utilities.ros.qos_profiles import NORMAL_QOS, RELIABLE_QOS
from std_msgs.msg import Bool

class StartNode(Node):
    def __init__(self):
        super().__init__('start_node')

        self.publisher_start = self.create_publisher(
            Bool,
            '/stl_mapping/start',
            RELIABLE_QOS
        )

        # publish true once to all listeners (that should already be running)
        self.publisher_start.publish(Bool(data=True))

def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    start_node = StartNode()
    rclpy.spin(start_node)

    start_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()