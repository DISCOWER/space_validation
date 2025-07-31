#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    """Launch Gazebo with two freeflyers running PX4 communicating over ROS 2."""
    model_arg = DeclareLaunchArgument("model", default_value="atmos")
    namespace_arg = DeclareLaunchArgument("namespace", default_value="snap")
    model = LaunchConfiguration("model")
    namespace = LaunchConfiguration("namespace")

    ld = LaunchDescription([model_arg, namespace_arg])

    # Run the Gazebo simulator and the PX4 SITL simulation. We add a delay
    # to the second robot to ensure they spawn in the same gazebo instance
    # run the px4_1.launch.py script twice
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [get_package_share_directory('discower_launch'), '/px4.launch.py']),
        # launch_arguments={'id':'0', 'pose':'1,0,0', 'name':'snap', 'delay':'0', 'model':'atmos'}.items()
        launch_arguments={'id':'0', 'pose':'1,0,0', 'name':{namespace}, 
                          'delay':'0', 'model':model, 'use_odom_bridge':'true'}.items()
    ))

    # Visualizer nodes which subscribe to the PX4 topics and converts them to sensible
    # topics for rviz
    ld.add_action(Node(
            package='px4_offboard',
            namespace=namespace,
            executable='visualizer',
            name='visualizer_0'
    )),

    # Rviz while loading a config file (valid for all three spacecrafts)
    ld.add_action(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', [os.path.join(get_package_share_directory('stl_mapping'), 'config.rviz')]]
    ))

    # # microros
    # ld.add_action(ExecuteProcess(
    #     cmd=["micro-xrce-dds-agent", "udp4", "-p", "8888"], output="screen",
    # ))

#     # Plotjuggler
#     ld.add_action(Node(
#             package='plotjuggler',
#             executable='plotjuggler',
#             name='plotjuggler',
#             arguments=['-l', os.path.join(get_package_share_directory('stl_mapping'), 'juggler_sitl_3.xml')]
#     ))

    return ld