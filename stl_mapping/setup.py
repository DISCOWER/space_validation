from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'stl_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join('config', '*.rviz'))),     # rviz configs
        (os.path.join('share', package_name), glob(os.path.join('config', '*.xml'))),      # plotjuggler configs

        # launch files
        (os.path.join('share', package_name), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),

        # data files (csv of motion plan)
        (os.path.join('share', package_name), glob(os.path.join('Planning/solutions', '*.npz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jorisv',
    maintainer_email='jorisv@kth.se',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                # planning and missions
                'main_planner = stl_mapping.planners.main_planner:main',
                'replanner = stl_mapping.planners.replanner:main',
                'scenario = stl_mapping.scenario:main',
                'reset = stl_mapping.reset:main',

                # controllers
                'ff_rate_mpc = stl_mapping.ff_rate_mpc:main',
                'ff_rate_mpc_impact = stl_mapping.ff_rate_mpc_impact:main',
                'ff_wrench_mpc_impact = stl_mapping.ff_wrench_mpc_impact:main',
                'ff_rate_mpc_velocity_keeping = stl_mapping.ff_rate_mpc_velocity_keeping:main',
                'ff_rate_qp_velocity_keeping = stl_mapping.ff_rate_qp_velocity_keeping:main',

                # helpers
                'impact_detector = stl_mapping.helpers.impact_detector:main',
                'odom_to_vehicle_local_position = stl_mapping.helpers.odom_to_vehicle_local_position:main',
                'odom_to_vehicle_angular_velocity = stl_mapping.helpers.odom_to_vehicle_angular_velocity:main',
                'odom_to_vehicle_attitude = stl_mapping.helpers.odom_to_vehicle_attitude:main',

                # test
                'test_replanner = stl_mapping.tests.test_replanner:main',
        ],
    },
)
