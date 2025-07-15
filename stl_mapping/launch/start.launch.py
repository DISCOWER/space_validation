#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    #! If errors occur, run this on the mo-cap PC 
    #! recording the cameras seems to cause issues

    # robots = ['snap', 'crackle']
    # robots = ['pop']
    robot_name = 'snap'
    topics_to_record = []
    topics_to_record.append(f'/{robot_name}/fmu/out/vehicle_local_position')
    topics_to_record.append(f'/{robot_name}/odom')
    topics_to_record.append(f'/{robot_name}/stl_mapping/entire_path')
    # topics_to_record.append(f'/{robot_name}/stl_mapping/reference_path')
    # topics_to_record.append(f'/{robot_name}/stl_mapping/predicted_path')
    # topics_to_record.append(f'/{robot_name}/fmu/in/vehicle_rates_setpoint')

    # record camera?
    # topics_to_record += ['/regions/goal_region']
    # topics_to_record += ['/camera/camera/color/image_raw']
    
    # record all topics?
    # topics_to_record = ['-a']

    record_cmd = ['ros2','bag','record']+topics_to_record

    ld = LaunchDescription()
    ld.add_action(ExecuteProcess(cmd=record_cmd))

    # Send a "Start" signal to anyone that cares
    ld.add_action(ExecuteProcess(
        cmd=[
            'ros2 topic pub --once /stl_mapping/start \
                std_msgs/msg/Bool \
                "{data: true}"'
        ], shell=True)
    )

    return ld
