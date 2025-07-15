import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import casadi as cs
import os
from Utilities.qos_profiles import NORMAL_QOS, RELIABLE_QOS
from Utilities.get_reference_trajectory import get_reference_trajectory, ReferenceTrajectory
        
from std_msgs.msg import Bool
from px4_msgs.msg import VehicleStatus, VehicleAttitude, VehicleAngularVelocity, VehicleLocalPosition
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint, OffboardControlMode

class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')

        # Subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status',
            self.vehicle_status_callback,
            NORMAL_QOS)
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            'fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            NORMAL_QOS)
        self.angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            'fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            NORMAL_QOS)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            NORMAL_QOS)
        self.start_sub = self.create_subscription(
            Bool, '/stl_mapping/start', self.start_callback, RELIABLE_QOS)

        # Publishers
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', NORMAL_QOS)
        self.publisher_torque_setpoint = self.create_publisher(VehicleTorqueSetpoint, 'fmu/in/vehicle_torque_setpoint', NORMAL_QOS)
        self.publisher_thrust_setpoint = self.create_publisher(VehicleThrustSetpoint, 'fmu/in/vehicle_thrust_setpoint', NORMAL_QOS)
        

        # Create the MPC solver and create timer callback to solve
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        # Load the plan from a file (declared as launch argument)
        self.plan_path = self.declare_parameter('plan_path', 'sp_solution_quat.npz').value
        plan_solution = np.load(self.plan_path)
        self.reference = ReferenceTrajectory(
            r=plan_solution['r'],
            dr=plan_solution['dr'],
            q=plan_solution['q'],
            dt=plan_solution['dt']
        )
        self.get_logger().info(f"Loaded plan from {self.plan_path}")

        # Initialize variables
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.started = False
        
        self.t0 = np.inf

    def vehicle_attitude_callback(self, msg):
            # NED-> ENU transformation
            # Receives quaternion in NED frame as (qw, qx, qy, qz)
            q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
            q_enu /= np.linalg.norm(q_enu)
            self.vehicle_attitude = q_enu.astype(float)

    def vehicle_local_position_callback(self, msg):
        # NED-> ENU transformation
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def vehicle_angular_velocity_callback(self, msg):
        # NED-> ENU transformation
        self.vehicle_angular_velocity[0] = msg.xyz[0]
        self.vehicle_angular_velocity[1] = -msg.xyz[1]
        self.vehicle_angular_velocity[2] = -msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def start_callback(self, msg):
        if msg.data:
            self.get_logger().info("Received start signal, beginning MPC computation.")
            # Start the MPC computation
            self.started = True
            self.t0 = time.time()
        else:
            self.get_logger().info("Received stop signal, stopping MPC computation.")

    def publish_offboard_mode(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        offboard_msg.body_rate = True   # rate control
        self.publisher_offboard_mode.publish(offboard_msg)

    def cmdloop_callback(self):
        t = time.time()

        self.publish_offboard_mode()


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    