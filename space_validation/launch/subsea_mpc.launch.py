#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    """Launch Gazebo with two freeflyers running PX4 communicating over ROS 2."""
    model_arg = DeclareLaunchArgument("model", default_value="bluerov")
    namespace_arg = DeclareLaunchArgument("namespace", default_value="itrl_rov_1")
    model = LaunchConfiguration("model")
    namespace = LaunchConfiguration("namespace")

    ld = LaunchDescription([model_arg, namespace_arg])

    # Visualizer nodes which subscribe to the PX4 topics and converts them to sensible
    # topics for rviz
    ld.add_action(Node(
            package='space_validation',
            namespace=namespace,
            executable='mpc_node',
            name='mpc_node_0',
            parameters=[{'x_offset': 1.2, 'y_offset': 0.0, 'z_offset': 1.25}, # exp1
                        # {'x_offset': 0.65, 'y_offset': 0.2, 'z_offset': 1.70}, # exp2
                        {'rate': 20.0},
                        {'model_name': model},
                        {'offset_free': True},
                        {'feedback_equivalence': True},
                        {'experiment': 'exp_1'}] # or 'exp_2'
    ))

    return ld