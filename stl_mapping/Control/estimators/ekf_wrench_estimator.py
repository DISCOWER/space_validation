import numpy as np
import casadi as cs
from Utilities.Robots import FreeFlyer
from Utilities.smarc_modelling.src.smarc_modelling.vehicles.BlueROV import BlueROV
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node

class EKFWrenchEstimator(Node):
    def __init__(self, dt:float = 0.1, robot_name:str = 'atmos'):
        super().__init__('ekf_wrench_estimator')
        self.robot_name = robot_name
        if robot_name == 'atmos':
            self.robot = FreeFlyer()
        elif robot_name == 'bluerov':
            self.robot = BlueROV()
        else:
            raise ValueError(f"Unknown robot name: {robot_name}")
        self.dt = dt

        # Create the Jacobian F (square matrix of state [v,w,fd,td])
        x = cs.SX.sym('x', 13)
        u = cs.SX.sym('u', 6)
        fd = cs.SX.sym('fd', 3)
        td = cs.SX.sym('td', 3)
        dx = self.robot.calculate_disturbed_dynamics(x,u,fd,td)
        x_kp1 = x + dx*self.dt
        # Jacobian
        F = cs.jacobian(cs.vertcat(x_kp1[7:13], fd, td), cs.vertcat(x[7:13], fd, td))
        # Into a symbolic function
        self.F = cs.Function('F', [x, u, fd, td], [F])

        self.mass = self.robot.m
        self.inertia = self.robot.inertia
        self.inertia_inv = np.linalg.inv(self.inertia)

        # captures how quickly it can change per second 
        # if self.robot_name == 'atmos':
        #     qv = (0.6/self.mass * self.dt)**2
        #     qw = [(0.12 / self.inertia[i, i] * self.dt)**2 for i in range(3)]
        #     qfd = (0.2*self.dt)**2      # 200mN/s
        #     qtd = (0.05*self.dt)**2
        #     rv = (1e-2*self.dt)**2
        #     rw = (0.05*self.dt)**2
        # elif self.robot_name == 'bluerov':
        qv = (0.6/self.mass * self.dt)**2
        qw = [(0.12 / self.inertia[i, i] * self.dt)**2 for i in range(3)]
        qfd = 15*(0.2*self.dt)**2      # 200mN/s
        qtd = 1*(0.05*self.dt)**2
        rv = (1e-2*self.dt)**2
        rw = (0.05*self.dt)**2
        # else:
        #     raise ValueError(f"Unknown robot name: {robot_name}")

        # State: [v, w, fd, td] in R^12
        self.x = np.zeros(12)  # Initial state vector
        self.P = 5*np.eye(12) # initial uncertainty
        self.Q = np.diag([qv]*3 + qw + [qfd]*3 + [qtd]*3) # Process noise covariance
        self.R = np.diag([rv]*3 + [rw]*3) # Measurement noise covariance

    def predict(self, x_meas, F_cmd, T_cmd):
        v = self.x[0:3]
        w = self.x[3:6]
        fd = self.x[6:9]    # Force disturbance in inertial frame
        td = self.x[9:12]   # Torque disturbance in body frame

        # System dynamics
        # Rot = R.from_quat(x_meas[3:7],scalar_first=True)
        # F_tot = F_cmd + R.inv().apply(fd)   # Total force in body frame
        # T_tot = T_cmd + td                  # Total torque in body frame

        # a_lin = F_tot / self.mass
        # a_ang = self.inertia_inv @ (T_tot - np.cross(w, self.inertia @ w))
        
        u = np.hstack((F_cmd, T_cmd))
        dx = self.robot.calculate_disturbed_dynamics(x_meas,u,fd,td)
        a_lin = dx[7:10]
        a_ang = dx[10:13]

        # Euler integration
        v_new = v + a_lin * self.dt
        w_new = w + a_ang * self.dt

        self.x = np.hstack((v_new, w_new, fd, td))

        # Jacobian matrix F
        F = self.F(x_meas, u, fd, td)
        F = np.array(F)

        self.P = F @ self.P @ F.T + self.Q

    def update(self, x_meas):
        # Measurement is [v, w] directly
        v_meas, w_meas = x_meas[7:10], x_meas[10:13]
        z = np.hstack((v_meas, w_meas))

        # Measurement Jacobian H
        H = np.zeros((6, 12))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - (H @ self.x)
        self.x = self.x + K @ y

        # self.get_logger().info(f"self.x: {self.x}")
        # self.get_logger().info(f"z: {z}")
        self.P = (np.eye(12) - K @ H) @ self.P

    def step(self, x_meas, F_cmd, T_cmd):
        self.predict(x_meas, F_cmd, T_cmd)
        self.update(x_meas)

        fd = self.x[6:9]
        td = self.x[9:12]

        return fd, td, self.P[6:12, 6:12]