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
        
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Bool
from px4_msgs.msg import VehicleStatus, VehicleAttitude, VehicleAngularVelocity, VehicleLocalPosition
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint, OffboardControlMode, ActuatorMotors

from Control.controllers.mpc_wrench import MpcWrench
from Control.estimators.ekf_wrench_estimator import EKFWrenchEstimator
from Utilities.rotations import quat_to_euler_np, q_to_rot_mat_np
from Utilities.rotations import enu_to_ned, ned_to_enu

class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')
        self.get_logger().info("Initializing MPC Node")

        self.mpc = MpcWrench()
        self.control = np.zeros((self.mpc.nu, 1))
        self.get_logger().info("MPC solver initialized successfully")

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
            Bool, 
            '/stl_mapping/start', 
            self.start_callback, 
            RELIABLE_QOS)
        self.actuator_motors_sub = self.create_subscription(
            ActuatorMotors,
            'fmu/out/actuator_motors',
            self.actuator_motors_callback,
            NORMAL_QOS)
        
        self.get_logger().info("MPC subscribers initialized successfully")

        # Publishers
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, 
            'fmu/in/offboard_control_mode', 
            NORMAL_QOS)
        self.publisher_torque_setpoint = self.create_publisher(
            VehicleTorqueSetpoint, 
            'fmu/in/vehicle_torque_setpoint', 
            NORMAL_QOS)
        self.publisher_thrust_setpoint = self.create_publisher(
            VehicleThrustSetpoint, 
            'fmu/in/vehicle_thrust_setpoint', 
            NORMAL_QOS)
        self.publisher_thrust_setpoint = self.create_publisher(
            VehicleThrustSetpoint,
            'fmu/in/vehicle_thrust_setpoint',
            NORMAL_QOS)
        self.publisher_torque_setpoint = self.create_publisher(
            VehicleTorqueSetpoint,
            'fmu/in/vehicle_torque_setpoint',
            NORMAL_QOS)
        self.predicted_path_pub = self.create_publisher(
            Path,
            'stl_mapping/predicted_path',
            10)
        self.reference_path_pub = self.create_publisher(
            Path,
            "stl_mapping/reference_path",
            10)
        self.reference_point_pub = self.create_publisher(
            PoseStamped,
            "stl_mapping/reference_point",
            10)
        self.entire_path_pub = self.create_publisher(
            Path,
            'stl_mapping/entire_path',
            10)
        self.force_torque_app_pub = self.create_publisher(
            WrenchStamped,
            "force_torque_app",
            10)
        self.disturbance_est_pub = self.create_publisher(
            WrenchStamped,
            'disturbance_estimate',
            10)
        
        self.get_logger().info("MPC publishers initialized successfully")

        # Disturbance variables
        self.offset_free = True
        self.F_app = np.zeros((3, 1))  # Force applied
        self.T_app = np.zeros((3, 1))  # Torque applied
        self.ekf_estimator = EKFWrenchEstimator(dt=0.05)
        self.fd_est = np.zeros(3)  # Estimated force disturbance
        self.td_est = np.zeros(3)  # Estimated torque disturbance

        # Create the MPC solver and create timer callback to solve
        timer_period = 0.2 # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        timer_period_offboard = 0.1 # seconds
        self.timer_offboard = self.create_timer(timer_period_offboard, self.publish_offboard_mode)

        timer_period_dist_est = self.ekf_estimator.dt # seconds
        self.timer_dist_est = self.create_timer(timer_period_dist_est, self.disturbance_estimation_callback)

        # Load the plan from a file (declared as launch argument)
        self.plan_path = self.declare_parameter('plan_path', 'sp_solution_bezier.npz').value

        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(this_file_dir, '../../../../share/stl_mapping/'))
        # path = '/home/none/space_ws/src/stl_mapping/stl_mapping/Planning/solutions/'
        self.plan_path = os.path.join(path, self.plan_path)
        self.get_logger().info(f"Loaded plan from {self.plan_path}")

        print(f"Loading plan from {self.plan_path}")
        plan_solution = np.load(self.plan_path)
        self.reference = ReferenceTrajectory(
            r=plan_solution['r'],
            dr=plan_solution['dr'],
            q=plan_solution['q'],
            dt=plan_solution['dt']
        )
        self.get_logger().info(f"q: {self.reference.q}")

        # Initialize variables
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.started = False
        
        self.t0 = np.inf
        self.get_logger().info("MPC Node initialized successfully")

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

    def actuator_motors_callback(self, msg: ActuatorMotors):
        B_F = 1.4 * np.array([
            [1., -1., 1., -1., 0., 0., 0., 0.],
            [0., 0., 0., 0., -1., 1., -1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0.]
            ])
        B_T = 1.4 * 0.12 * np.array([
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [-1., 1., 1., -1., -1., 1., 1., -1.]
            ])
        self.F_app = B_F @ np.array(msg.control[0:8]).reshape(8, 1)
        self.T_app = B_T @ np.array(msg.control[0:8]).reshape(8, 1)

        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'map'

        wrench_msg.wrench.force.x = float(self.F_app[0])
        wrench_msg.wrench.force.y = float(self.F_app[1])
        wrench_msg.wrench.force.z = float(self.F_app[2])
        wrench_msg.wrench.torque.x = float(self.T_app[0])
        wrench_msg.wrench.torque.y = float(self.T_app[1])
        wrench_msg.wrench.torque.z = float(self.T_app[2])

        self.force_torque_app_pub.publish(wrench_msg)

    def start_callback(self, msg):
        if msg.data:
            self.get_logger().info("Received start signal, beginning MPC computation.")
            # Start the MPC computation
            self.started = True
            self.t0 = time.time()
        else:
            self.get_logger().info("Received stop signal, stopping MPC computation.")

    def publish_estimated_disturbance(self, fd_est, td_est):
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'map'

        wrench_msg.wrench.force.x = float(fd_est[0])
        wrench_msg.wrench.force.y = float(fd_est[1])
        wrench_msg.wrench.force.z = float(fd_est[2])
        wrench_msg.wrench.torque.x = float(td_est[0])
        wrench_msg.wrench.torque.y = float(td_est[1])
        wrench_msg.wrench.torque.z = float(td_est[2])

        self.disturbance_est_pub.publish(wrench_msg)

    def publish_wrench_setpoint(self, u):
        force_output_msg = VehicleThrustSetpoint()
        force_output_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        torque_output_msg = VehicleTorqueSetpoint()
        torque_output_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        # ENU -> NED transformation
        force_output_msg.xyz = [u[0]*0.5, -u[1]*0.5, -u[2]*0.5]
        torque_output_msg.xyz = [u[3]*0.5, -u[4]*0.5, -u[5]*0.5]
        # EKF in FLU frame so don't transform
        # self.F_app = u[:3]
        # self.T_app = u[3:6]

        self.publisher_thrust_setpoint.publish(force_output_msg)
        self.publisher_torque_setpoint.publish(torque_output_msg)

    def publish_offboard_mode(self):
        # self.get_logger().info("Publishing offboard mode")
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        offboard_msg.thrust_and_torque = True
        self.publisher_offboard_mode.publish(offboard_msg)
    
    def vector2PoseMsg(self, frame_id, position, attitude):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.orientation.w = attitude[0]
        pose_msg.pose.orientation.x = attitude[1]
        pose_msg.pose.orientation.y = attitude[2]
        pose_msg.pose.orientation.z = attitude[3]
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        return pose_msg

    def publish_predicted_path(self, x_pred, current_attitude):
        now = self.get_clock().now()
        predicted_path_msg = Path()
        predicted_path_msg.header.stamp = now.to_msg()
        predicted_path_msg.header.frame_id = 'map'

        for i, predicted_state in enumerate(x_pred):
            # Calculate future time offset
            future_time = now + rclpy.duration.Duration(seconds=i * self.mpc.dt)

            # Create PoseStamped
            pose_stamped = self.vector2PoseMsg('map', predicted_state[0:3], current_attitude)
            pose_stamped.header.stamp = future_time.to_msg()
            pose_stamped.header.frame_id = 'map'

            predicted_path_msg.poses.append(pose_stamped)

        self.predicted_path_pub.publish(predicted_path_msg)
    
    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        pub.publish(msg)

    def disturbance_estimation_callback(self):
        x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
        FT = np.concatenate((self.F_app, self.T_app), axis=0)
        self.fd_est, self.td_est = self.ekf_estimator.step(x0.flatten(), FT.flatten()) 
        # self.fd_est = np.zeros_like(self.fd_est)
        # self.td_est = np.zeros_like(self.td_est)
        self.publish_estimated_disturbance(self.fd_est, self.td_est)

    def cmdloop_callback(self):
        # self.get_logger().info("Command loop callback")
        t = time.time()

        x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)

        if not self.started:
            # self.get_logger().info("Mission not started, using constant reference (t=0)")
            times = np.zeros(self.mpc.Nx + 1)  # No time since start, constant reference
        else:
            # self.get_logger().info("Mission started, calculating reference trajectory")
            t_mpc = t - self.t0
            times = np.linspace(t_mpc, t_mpc + self.mpc.Nx * self.mpc.dt, self.mpc.Nx + 1)
            # times = np.arange(t_mpc, t_mpc + self.mpc.Nx * self.mpc.dt, self.mpc.dt)
            # self.get_logger().info(f"t_mpc: {t_mpc}, times: {times}")

        if self.offset_free:
            # rotation matrix of FRU in ENU
            rotmat = q_to_rot_mat_np(self.vehicle_attitude)
            u_ref = -np.concatenate((rotmat.transpose() @ self.fd_est.reshape(3, 1), self.td_est.reshape(3, 1)), axis=0)

        x_ref = np.zeros((13, self.mpc.Nx + 1))  # Initialize reference trajectory
        for idx, ti in enumerate(times):
            # x_ref[:, idx] = np.array([1., 0., 0.,
            #                           0., 0., 0., 1.,
            #                           0., 0., 0.,
            #                           0., 0., 0.]).reshape(13,)
            x_ref[:, idx] = get_reference_trajectory(ti, self.reference, order='zyx')

        x_ref_u = np.tile(u_ref if self.offset_free else np.zeros((6, 1)), (1, x_ref.shape[1]))

        # x_ref contains reference in order p q dp dq, convert to order p, dp, q, dq
        x_ref = np.concatenate((
            x_ref[0:3, :],      # Position [ENU]
            x_ref[7:10, :],     # Linear velocity [ENU]
            x_ref[3:7, :],      # Quaternion [FLU in ENU]
            x_ref[10:13, :],    # Angular velocity [FLU]
            x_ref_u
        ))
        self.get_logger().info(f"euler: {quat_to_euler_np(x_ref[6:10, 0], order='zyx')}")
        self.get_logger().info(f"x_ref: {x_ref[:, 0].flatten()}")
        # self.get_logger().info(f"x0: {x0.flatten()}")

        # Get control input
        if self.offset_free:
            self.control, x_pred = self.mpc.get_input(x0, x_ref, fd=self.fd_est, td=self.td_est)
        else:
            self.control, x_pred = self.mpc.get_input(x0, x_ref)
        # print(f"Control: {self.control.flatten()}")

        # quat_error = (x_pred[0, 6:10] @ x_ref[6:10, 0])**2
        # self.get_logger().warning(f"quat_error: {1-quat_error}")
        # pos_error = np.linalg.norm(x_pred[0, 0:2] - x_ref[0:2, 0])
        # self.get_logger().warning(f"pos_error: {pos_error}")

        # Publish the reference and predicted path for rviz
        setpoint_path_msg = Path()
        for idx in range(x_ref.shape[1]):
            setpoint = x_ref[:, idx]
            setpoint_pose_msg = self.vector2PoseMsg('map', setpoint[0:3], setpoint[6:10])
            setpoint_path_msg.header = setpoint_pose_msg.header
            setpoint_path_msg.poses.append(setpoint_pose_msg)
        self.reference_path_pub.publish(setpoint_path_msg)


        self.reference_point_pub.publish(
            self.vector2PoseMsg('map', x_ref[0:3, 0], x_ref[6:10, 0])
        )

        x_pred = x_pred.T
        predicted_path_msg = Path()
        for idx in range(x_pred.shape[1]):
            predicted_state = x_pred[:, idx]
            predicted_pose_msg = self.vector2PoseMsg('map', predicted_state[0:3], predicted_state[6:10])
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_wrench_setpoint(self.control)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    