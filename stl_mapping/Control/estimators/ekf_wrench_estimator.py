import numpy as np
from Utilities.Robots import FreeFlyer

class EKFWrenchEstimator:
    def __init__(self, dt:float = 0.1, robot: FreeFlyer = None):
        self.robot = robot if robot is not None else FreeFlyer()

        self.mass = self.robot.mass
        self.inertia = self.robot.inertia
        self.inertia_inv = np.linalg.inv(self.inertia)

        self.dt = dt
        qv = (0.6/self.mass * self.dt)**2
        qw = [(0.12 / self.inertia[i, i] * self.dt)**2 for i in range(3)]
        qfd = (0.2*self.dt)**2
        qtd = (0.05*self.dt)**2
        rv = (1e-2*self.dt)**2
        rw = (0.05*self.dt)**2

        # State: [v, w, fd, td] in R^12
        self.x = np.zeros(12)  # Initial state vector
        self.P = 100*np.eye(12) # initial uncertainty
        self.Q = np.diag([qv]*3 + qw + [qfd]*3 + [qtd]*3) # Process noise covariance
        self.R = np.diag([rv]*3 + [rw]*3) # Measurement noise covariance

    def get_rotMat(self, q):
        """Returns rotation matrix from quaternion q = [w, x, y, z]."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2,     2*(x*y - w*z),         2*(x*z + w*y)],
            [2*(x*y + w*z),           1 - 2*x**2 - 2*z**2,   2*(y*z - w*x)],
            [2*(x*z - w*y),           2*(y*z + w*x),         1 - 2*x**2 - 2*y**2]
        ])

    def predict(self, q, F_cmd, T_cmd):
        v = self.x[0:3]
        w = self.x[3:6]
        fd = self.x[6:9]
        td = self.x[9:12]

        # System dynamics
        R = self.get_rotMat(q)
        F_tot = R @ F_cmd + fd
        T_tot = T_cmd + td

        a_lin = F_tot / self.mass
        a_ang = self.inertia_inv @ (T_tot - np.cross(w, self.inertia @ w))

        # #! If we want to use the robot model directly
        # x = self.x[0:13]
        # u = np.vstack((F_cmd, T_cmd))
        # dx = self.robot.calculate_disturbed_dynamics(x,u,fd,td)
        # a_lin = dx[7:10]
        # a_ang = dx[10:13]

        # Euler integration
        v_new = v + a_lin * self.dt
        w_new = w + a_ang * self.dt

        self.x = np.hstack((v_new, w_new, fd, td))

        # Jacobian matrix F
        F = np.eye(12)
        F[0:3, 6:9] = np.eye(3) * (self.dt / self.mass)
        F[3:6, 9:12] = self.inertia_inv * self.dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, v_meas, w_meas):
        # Measurement is [v, w] directly
        z = np.hstack((v_meas, w_meas))

        # Measurement Jacobian H
        H = np.zeros((6, 12))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - (H @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(12) - K @ H) @ self.P

    def step(self, x_meas, F_cmd, T_cmd):
        q_meas = x_meas[3:7]
        v_meas = x_meas[7:10]
        w_meas = x_meas[10:13]

        self.predict(q_meas, F_cmd, T_cmd)
        self.update(v_meas, w_meas)

        fd = self.x[6:9]
        td = self.x[9:12]

        return fd, td, self.P[6:12, 6:12]