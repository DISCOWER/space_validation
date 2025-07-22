import numpy as np

class EKFWrenchEstimator:
    def __init__(self):
        self.mass = 16.8 + 1.0  # kg
        self.inertia = np.diag([0.297 * self.mass / 16.8] * 3)
        self.inertia_inv = np.linalg.inv(self.inertia)

        # State: [fd, td] in R^6
        self.x = np.zeros(6)
        self.P = np.eye(6) # initial uncertainty
        # self.Q = 0.1 * np.diag([0.05**2]*3 + [0.001**2]*3) # process noise covariance
        # self.R = 10 * np.diag([0.001]*3 + [0.001]*3)
        self.Q = np.diag([0.001]*6)
        self.R = np.diag([0.01]*6)

        self.v_prev = None
        self.w_prev = None
        self.first_step = True
        self.t_prev = None
        self.t = None

        self.x_meas = None
        self.u = None

    def get_rotMat(self, q):
        """Returns rotation matrix from quaternion q = [w, x, y, z]."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2,     2*(x*y - w*z),         2*(x*z + w*y)],
            [2*(x*y + w*z),           1 - 2*x**2 - 2*z**2,   2*(y*z - w*x)],
            [2*(x*z - w*y),           2*(y*z + w*x),         1 - 2*x**2 - 2*y**2]
        ])

    def predict(self):
        self.P = self.P + self.Q  # x_k+1 = x_k + w_k (random walk)

    def update(self, accel_meas):
        fd = self.x[0:3]
        td = self.x[3:6]

        q = self.x_meas[6:10]
        w = self.x_meas[10:13]
        F_cmd = self.u[:3]
        T_cmd = self.u[3:6]

        R = self.get_rotMat(q)
        RT = R.T

        # Predicted measurements
        F_tot = R @ F_cmd + fd
        T_tot = T_cmd + RT @ td

        a_lin = F_tot / self.mass
        a_ang = self.inertia_inv @ (T_tot - np.cross(w, self.inertia @ w))
        z_pred = np.concatenate([a_lin, a_ang])

        # Jacobian
        H = np.zeros((6, 6))
        H[0:3, 0:3] = np.eye(3) / self.mass
        H[3:6, 3:6] = self.inertia_inv @ RT

        # Kalman update
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = accel_meas - z_pred
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def step(self, x_meas, u, t):
        if self.t_prev is None:
            self.t_prev = t
            dt = 0.1
        else:
            dt = t - self.t_prev
            self.t_prev = t
            
        dt *= 1e-9  # Convert nanoseconds to seconds

        v = x_meas[3:6]
        w = x_meas[10:13]

        if self.first_step:
            self.v_prev = v
            self.w_prev = w
            self.first_step = False
            return np.zeros(3), np.zeros(3)

        v_dot = (v - self.v_prev) / dt
        w_dot = (w - self.w_prev) / dt
        accel_meas = np.concatenate([v_dot, w_dot])

        self.v_prev = v
        self.w_prev = w

        self.x_meas = x_meas
        self.u = u

        self.predict()
        self.update(accel_meas)

        fd = self.x[0:3]
        td = self.x[3:6]
        return fd, td