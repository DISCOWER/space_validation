#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

import time
from pathlib import Path
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    #! If errors occur, run this on the mo-cap PC 
    #! recording the cameras seems to cause issues
    HOME = str(Path.home())

    ld = LaunchDescription()

    # Get the current date and time in a yyyy_mm_dd-hh_mm_ss string format
    # so we can also copy planning files to the same rosbag!
    current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    rosbag_name = f'space_validation_run_{current_time}/'

    topics_to_record = ['-a']

    #! Record the specified topics
    record_cmd = ['ros2','bag','record']+topics_to_record#+['-o', rosbag_name]
    record_proc = ExecuteProcess(cmd=record_cmd)

    #! Send a "Start" signal to anyone that cares
    start_proc = Node(
        package='space_validation',
        executable='start_node'
    )

    #! Copy the planning files to the rosbag directory
    copy_proc = ExecuteProcess(
        cmd=[
            f'cp -r {HOME}/space_validation_ws/src/space_validation/Planning/solutions \
                {HOME}/space_validation_ws/{rosbag_name}/'
        ], shell=True)
    
    # Add it all together
    ld.add_action(start_proc)
    ld.add_action(record_proc)
    ld.add_action(RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=record_proc,
            on_start=[TimerAction(period=0.5, actions=[copy_proc])]
        )
    ))
    return ld
