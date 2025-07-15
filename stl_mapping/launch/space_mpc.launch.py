#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    """Launch Gazebo with two freeflyers running PX4 communicating over ROS 2."""
    ld = LaunchDescription()

    # Visualizer nodes which subscribe to the PX4 topics and converts them to sensible
    # topics for rviz
    ld.add_action(Node(
            package='stl_mapping',
            namespace='snap',
            executable='mpc_node',
            name='mpc_node_0'
    ))

    return ld