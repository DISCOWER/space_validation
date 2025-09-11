import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
import os
from Utilities.ros.qos_profiles import NORMAL_QOS, RELIABLE_QOS
from Utilities.get_reference_trajectory import get_reference_trajectory, ReferenceTrajectory
        
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, WrenchStamped, TwistWithCovarianceStamped
from std_msgs.msg import Bool
from px4_msgs.msg import VehicleStatus, VehicleAttitude, VehicleAngularVelocity, VehicleLocalPosition
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint, OffboardControlMode, ActuatorMotors

from Control.controllers.mpc_wrench import MpcWrench
from Control.controllers.mpc_fbl_wrench import MpcFBLWrench
from Control.estimators.ekf_wrench_estimator import EKFWrenchEstimator
from Utilities.rotations import quat_to_euler_np
from Utilities.Robots import FreeFlyer
from Utilities.smarc_modelling.src.smarc_modelling.vehicles.BlueROV import BlueROV
import copy

class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')
        self.get_logger().info("Initializing MPC Node")

        self.model_name = self.declare_parameter('model_name', 'atmos').value
        self.other_model_name = 'bluerov' if self.model_name == 'atmos' else 'atmos'

        self.x_offset = self.declare_parameter('x_offset', 0.0).value
        self.y_offset = self.declare_parameter('y_offset', 0.0).value
        self.z_offset = self.declare_parameter('z_offset', 0.0).value
        self.offset = np.array([self.x_offset, self.y_offset, self.z_offset])
        self.rate = self.declare_parameter('rate', 5.0).value

        if self.model_name == "atmos" or self.model_name == "cubesat":
            self.robot = FreeFlyer()
        elif self.model_name == "bluerov":
            self.robot = BlueROV()
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        self.mpc = MpcWrench(model_name=self.model_name)
        self.fbl_mpc = MpcFBLWrench(model_name=self.model_name)
        self.control = np.zeros((self.mpc.nu, 1))
        self.get_logger().info("MPC solver initialized successfully")

        # Subscribers
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            NORMAL_QOS)
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
        self.local_position_sub_v1 = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            NORMAL_QOS)
        self.local_position_sub_v1 = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position_v1',
            self.vehicle_local_position_callback,
            NORMAL_QOS)
        self.start_sub = self.create_subscription(
            Bool, 
            '/stl_mapping/start', 
            self.start_callback, 
            RELIABLE_QOS)
        # self.actuator_motors_sub = self.create_subscription(
        #     ActuatorMotors,
        #     'fmu/out/actuator_motors',
        #     self.actuator_motors_callback,
        #     NORMAL_QOS)
        
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
        self.force_torque_cmd_pub = self.create_publisher(
            WrenchStamped,
            "force_torque_cmd",
            10)
        self.disturbance_est_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            f'{self.model_name}/disturbance_estimate',
            10)
        self.other_disturbance_est_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            f'{self.other_model_name}/disturbance_estimate',
            10)
        self.reference_p_v_pub = self.create_publisher(
            VehicleLocalPosition,
            'stl_mapping/reference_pos_vel',
            10)
        self.reference_q_pub = self.create_publisher(
            VehicleAttitude,
            'stl_mapping/reference_q',
            10)
        self.reference_w_pub = self.create_publisher(
            VehicleAngularVelocity,
            'stl_mapping/reference_w',
            10)

        self.get_logger().info("MPC publishers initialized successfully")

        # Settings
        self.offset_free = True
        self.feedback_equivalence = True

        # Disturbance variables
        self.F_cmd = np.zeros((3, 1))  # Force commanded
        self.T_cmd = np.zeros((3, 1))  # Torque commanded
        self.other_F_cmd = np.zeros((3, 1))  # Force commanded
        self.other_T_cmd = np.zeros((3, 1))  # Torque commanded

        #! Temp
        self.other_control = np.zeros((6, 1))
        self.ekf_estimator = EKFWrenchEstimator(dt=0.02, robot_name=self.model_name)
        self.other_ekf_estimator = EKFWrenchEstimator(dt=self.ekf_estimator.dt, robot_name=self.other_model_name)
        self.fd_est = np.zeros(3)  # Estimated force disturbance
        self.td_est = np.zeros(3)  # Estimated torque disturbance
        self.dist_cov = np.zeros((6, 6))  # Disturbance covariance matrix
        self.other_fd_est = np.zeros(3)
        self.other_td_est = np.zeros(3)
        self.other_dist_cov = np.zeros((6, 6))

        # The PX4 uses normalized wrench input. If NORMALIZED_WRENCH is True, the control
        # input has to be normalized. I will at some point do a PR to the PX4 to
        # support non-normalized wrench input.
        NORMALIZED_WRENCH = True  # Use normalized wrench for control input
        self.F_thruster = 1.4  # Thrust force per motor
        self.r_thruster = 0.12  # Distance from center to thruster in meters
        if self.model_name == "atmos" or self.model_name == "cubesat":
            self.F_scaling = 2 * self.F_thruster if NORMALIZED_WRENCH else 1.0
            self.T_scaling = 4 * self.r_thruster * self.F_thruster if NORMALIZED_WRENCH else 1.0
        elif self.model_name == "bluerov":
            self.F_scaling = np.array([72, 72, 26])
            # self.T_scaling = np.array([12, 14, 22])
            self.T_scaling = np.array([20, 14, 22])
        else:
            raise Exception("Unknown model name for wrench scaling")

        # Create the MPC solver and create timer callback to solve
        timer_period = 1.0/self.rate # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        timer_period_offboard = 0.1 # seconds
        self.timer_offboard = self.create_timer(timer_period_offboard, self.publish_offboard_mode)

        timer_period_dist_est = self.ekf_estimator.dt # seconds
        self.timer_dist_est = self.create_timer(timer_period_dist_est, self.disturbance_estimation_callback)
        self.timer_other_dist_est = self.create_timer(timer_period_dist_est, self.other_disturbance_estimation_callback)

        # Load the plan from a file (declared as launch argument)
        self.plan_path = self.declare_parameter('plan_path', f'{self.model_name}_solution_bezier.npz').value

        exp = 'exp_2'
        path = os.path.abspath(os.path.join(os.path.expanduser("~"), f'space_ws/src/stl_mapping/stl_mapping/Planning/solutions/{exp}/'))
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
        self.vehicle_local_velocity_body = np.array([0.0, 0.0, 0.0])
        self.started = False
        
        self.t0 = np.inf
        self.get_logger().info("MPC Node initialized successfully")

        # # Publish the entire path
        # times = np.linspace(0, self.reference.r.shape[0]*self.reference.dt, 250)

    def vehicle_attitude_callback(self, msg):
        if self.model_name == "bluerov":
            self.vehicle_attitude = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)
        elif self.model_name == "atmos" or self.model_name == "cubesat":
            # if we're on atmos, we rotate the frame +90 degrees around z
            q = R.from_euler('z', -np.pi/2)
            q_msg = R.from_quat([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], scalar_first=True)  # wxyz
            q_total = q * q_msg
            self.vehicle_attitude = q_total.as_quat(scalar_first=True)  # wxyz
            # self.get_logger().info(f"q: {self.vehicle_attitude}")

    def vehicle_local_position_callback(self, msg):
        if self.model_name == "bluerov":
            self.vehicle_local_position = np.array([msg.x, msg.y, msg.z])
            self.vehicle_local_velocity = np.array([msg.vx, msg.vy, msg.vz])
            # get a body-frame velocity, because this is what the MPC model requires
            q = R.from_quat(self.vehicle_attitude,scalar_first=True)
            self.vehicle_local_velocity_body = q.inv().apply(self.vehicle_local_velocity)
        elif self.model_name == "atmos" or self.model_name == "cubesat":
            # if we're on atmos, we rotate the frame +90 degrees around z
            q = R.from_euler('z', -np.pi/2)
            self.vehicle_local_position = q.apply(np.array([msg.x, msg.y, msg.z]))
            self.vehicle_local_velocity = q.apply(np.array([msg.vx, msg.vy, msg.vz]))
            # get a body-frame velocity, because this is what the MPC model requires
            q = R.from_quat(self.vehicle_attitude,scalar_first=True)
            self.vehicle_local_velocity_body = q.inv().apply(self.vehicle_local_velocity)

    def vehicle_angular_velocity_callback(self, msg):
        self.vehicle_angular_velocity = np.array([msg.xyz[0], msg.xyz[1], msg.xyz[2]])

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

    def publish_estimated_disturbance(self, fd_est, td_est, dist_cov, other=False):
        wrench_msg = TwistWithCovarianceStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'map'

        wrench_msg.twist.twist.linear.x = float(fd_est[0])
        wrench_msg.twist.twist.linear.y = float(fd_est[1])
        wrench_msg.twist.twist.linear.z = float(fd_est[2])
        wrench_msg.twist.twist.angular.x = float(td_est[0])
        wrench_msg.twist.twist.angular.y = float(td_est[1])
        wrench_msg.twist.twist.angular.z = float(td_est[2])
        wrench_msg.twist.covariance = dist_cov.flatten().tolist() + [0.0] * (36 - len(dist_cov.flatten()))

        if other:
            self.other_disturbance_est_pub.publish(wrench_msg)
        else:
            self.disturbance_est_pub.publish(wrench_msg)

    def publish_wrench_setpoint(self, u):
        force_output_msg = VehicleThrustSetpoint()
        force_output_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        torque_output_msg = VehicleTorqueSetpoint()
        torque_output_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        # Scaling
        u[0:3] /= self.F_scaling
        u[3:6] /= self.T_scaling

        force_output_msg.xyz = [u[0], u[1], u[2]]
        torque_output_msg.xyz = [u[3], u[4], u[5]]

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
                        self.vehicle_attitude[0],
                        self.vehicle_attitude[1],
                        self.vehicle_attitude[2],
                        self.vehicle_attitude[3],
                        self.vehicle_local_velocity_body[0],
                        self.vehicle_local_velocity_body[1],
                        self.vehicle_local_velocity_body[2],
                        self.vehicle_angular_velocity[0],
                        self.vehicle_angular_velocity[1],
                        self.vehicle_angular_velocity[2]]).reshape(13, 1)
        self.fd_est, self.td_est, self.dist_cov = self.ekf_estimator.step(x0.flatten(), self.F_cmd.flatten(), self.T_cmd.flatten())
        self.publish_estimated_disturbance(self.fd_est, self.td_est, self.dist_cov)

    def other_disturbance_estimation_callback(self):
        x0 = np.array([self.vehicle_local_position[0],
                        self.vehicle_local_position[1],
                        self.vehicle_local_position[2],
                        self.vehicle_attitude[0],
                        self.vehicle_attitude[1],
                        self.vehicle_attitude[2],
                        self.vehicle_attitude[3],
                        self.vehicle_local_velocity_body[0],
                        self.vehicle_local_velocity_body[1],
                        self.vehicle_local_velocity_body[2],
                        self.vehicle_angular_velocity[0],
                        self.vehicle_angular_velocity[1],
                        self.vehicle_angular_velocity[2]]).reshape(13, 1)
        self.other_fd_est, self.other_td_est, self.other_dist_cov = self.other_ekf_estimator.step(x0.flatten(), self.other_F_cmd.flatten(), self.other_T_cmd.flatten())
        self.publish_estimated_disturbance(self.other_fd_est, self.other_td_est, self.other_dist_cov, other=True)

    def publish_reference_state(self, x_ref):
        p_v_msg = VehicleLocalPosition()
        p_v_msg.x = float(x_ref[0])
        p_v_msg.y = float(x_ref[1])
        p_v_msg.z = float(x_ref[2])
        p_v_msg.vx = float(x_ref[7])
        p_v_msg.vy = float(x_ref[8])
        p_v_msg.vz = float(x_ref[9])
        self.reference_p_v_pub.publish(p_v_msg)

        q_msg = VehicleAttitude()
        q_msg.q = [float(x_ref[3]), float(x_ref[4]), float(x_ref[5]), float(x_ref[6])]
        self.reference_q_pub.publish(q_msg)

        w_msg = VehicleAngularVelocity()
        w_msg.xyz = [float(x_ref[10]), float(x_ref[11]), float(x_ref[12])]
        self.reference_w_pub.publish(w_msg)

    def cmdloop_callback(self):
        # self.get_logger().info("Command loop callback")
        t = time.time()

        x0 = np.array([self.vehicle_local_position[0],
                       self.vehicle_local_position[1],
                       self.vehicle_local_position[2],
                       self.vehicle_attitude[0],
                       self.vehicle_attitude[1],
                       self.vehicle_attitude[2],
                       self.vehicle_attitude[3],
                       self.vehicle_local_velocity_body[0],
                       self.vehicle_local_velocity_body[1],
                       self.vehicle_local_velocity_body[2],
                       self.vehicle_angular_velocity[0],
                       self.vehicle_angular_velocity[1],
                       self.vehicle_angular_velocity[2]]).reshape(13, 1)

        # self.get_logger().info(f"x0: {x0.flatten()}")
        if not self.started:
            # self.get_logger().info("Mission not started, using constant reference (t=0)")
            times = np.zeros(self.mpc.Nx + 1)  # No time since start, constant reference
        else:
            # self.get_logger().info("Mission started, calculating reference trajectory")
            t_mpc = t - self.t0
            times = np.linspace(t_mpc, t_mpc + self.mpc.Nx * self.mpc.dt, self.mpc.Nx + 1)
            # self.get_logger().info(f"t_mpc: {t_mpc}, times: {times}")

        if self.offset_free:
            # limit fd_est and td_est to values between -10 and 10
            if self.feedback_equivalence:
                fd_est = np.clip(self.other_fd_est, -10, 10)
                td_est = np.clip(self.other_td_est, -5, 5)
            else:
                fd_est = np.clip(self.fd_est, -5, 5)
                td_est = np.clip(self.td_est, -1, 1)
            # rotation matrix of FRD in NED
            q = R.from_quat(self.vehicle_attitude, scalar_first=True)
            u_ref = -np.concatenate((q.inv().apply(fd_est), td_est), axis=0).reshape((6, 1))
            # self.get_logger().info(f"u_ref: {u_ref.flatten()}")
        else:
            u_ref = np.zeros((6, 1))
            fd_est = np.zeros((3, 1))
            td_est = np.zeros((3, 1))

        x_ref = np.zeros((13, self.mpc.Nx + 1))  # Initialize reference trajectory

        for idx, ti in enumerate(times):
        #     #! Easy sys-id
        #     # get the first index of times for which ti > times
        #     if t - self.t0 < 0:
        #         test_idx = 0
        #     else:
        #         test_idx = int((t-self.t0) // period) % test_points.shape[0]
        #     # self.get_logger().info(f"test_times: {test_times}")
        #     self.get_logger().info(f"idx: {test_idx}")
        #     self.get_logger().info(f"dt: {t-self.t0}")
        #     x_ref[:, idx] = test_points[test_idx]
            # self.get_logger().info(f"{x_ref[:, test_idx].flatten()}")
            # x_ref[:, idx] = np.array([1.5, 0.0, -0.6,
            #                            1.0, 0., 0., 0.,
            #                         #   0.5, 0.5, 0.5, 0.5,
            #                         #  1/np.sqrt(2), 0., 0., 1/np.sqrt(2),
            #                         # 1/np.sqrt(2), -1/np.sqrt(2), 0., 0.,
            #                           0., 0., 0.,
            #                           0., 0., 0.]).reshape(13,)
            x_ref[:, idx] = get_reference_trajectory(ti, self.reference, order='xyz')

        x_ref = np.vstack((x_ref, np.repeat(u_ref, x_ref.shape[1], axis=1)))  # Append u_ref to x_ref
        x_ref[:3, :] += self.offset.reshape(3, 1)

        # self.get_logger().info(f"euler: {quat_to_euler_np(x_ref[3:7, 0], order='xyz')}")
        # self.get_logger().info(f"x_ref: {x_ref[:3, 0].flatten()}")
        # self.get_logger().info(f"p: {x0[:3,0].flatten()}")
        # self.get_logger().info(f"p_ref: {x_ref[:3, 0].flatten()}")
        # self.get_logger().info(f"v: {x0[7:10,0]}")
        # self.get_logger().info(f"g_vec: {self.robot.calculate_test_g(x0)}")

        self.publish_reference_state(x_ref[:,0])

        # self.get_logger().info(f"fd_est: {fd_est}, td_est: {td_est}")
        # self.get_logger().info(f"u_ref: {u_ref.flatten()}")

        # Get control input
        if self.feedback_equivalence:
            if self.offset_free:
                self.control, x_pred, self.other_control, _ = self.fbl_mpc.get_input(x0, x_ref, fd=fd_est, td=td_est)
            else:
                self.control, x_pred, self.other_control, _ = self.fbl_mpc.get_input(x0, x_ref)
        else:
            if self.offset_free:
                self.control, x_pred = self.mpc.get_input(x0, x_ref, fd=fd_est, td=td_est)
            else:
                self.control, x_pred = self.mpc.get_input(x0, x_ref)
            # Run the other MPC as well, just for continuity and rosbags
            _, _, self.other_control, _ = self.fbl_mpc.get_input(x0, x_ref)

        # self.get_logger().info(f"Control: {self.control.flatten()}")
        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.F_cmd = copy.deepcopy(self.control[0:3])
            self.T_cmd = copy.deepcopy(self.control[3:6])
            self.other_F_cmd = copy.deepcopy(self.other_control[0:3])
            self.other_T_cmd = copy.deepcopy(self.other_control[3:6])
        else:
            self.F_cmd = np.zeros((3, 1))
            self.T_cmd = np.zeros((3, 1))
            self.other_F_cmd = np.zeros((3, 1))
            self.other_T_cmd = np.zeros((3, 1))

        self.get_logger().info(f"Control (F,N; T,Nm): {self.control[0:3].flatten()}, {self.control[3:6].flatten()}")

        # quat_error = (x_pred[0, 6:10] @ x_ref[6:10, 0])**2
        # self.get_logger().warning(f"quat_error: {1-quat_error}")
        # pos_error = np.linalg.norm(x_pred[0, 0:2] - x_ref[0:2, 0])
        # self.get_logger().warning(f"pos_error: {pos_error}")

        # Publish the reference and predicted path for rviz
        setpoint_path_msg = Path()
        for idx in range(x_ref.shape[1]):
            setpoint = x_ref[:, idx]
            setpoint_pose_msg = self.vector2PoseMsg('map', setpoint[0:3], setpoint[3:7])
            setpoint_path_msg.header = setpoint_pose_msg.header
            setpoint_path_msg.poses.append(setpoint_pose_msg)
        self.reference_path_pub.publish(setpoint_path_msg)


        self.reference_point_pub.publish(
            self.vector2PoseMsg('map', x_ref[0:3, 0], x_ref[3:7, 0])
        )

        x_pred = x_pred.T
        predicted_path_msg = Path()
        for idx in range(x_pred.shape[1]):
            predicted_state = x_pred[:, idx]
            predicted_pose_msg = self.vector2PoseMsg('map', predicted_state[0:3], predicted_state[3:7])
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_wrench_setpoint(self.control)


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    