import numpy as np
import casadi as cs
from Utilities.helpers import HyperRectangle


class Robot:
    def __init__(self,n_x:int,n_u:int):
        self.n_x = n_x  # Number of state variables
        self.n_u = n_u  # Number of control inputs
        self.x = np.zeros((n_x,))
        self.u = np.zeros((n_u,))

        self.fx = None  # State dynamics function
        self.gx = None

    def set_state(self, x:np.ndarray):
        assert x.shape==(self.n_x,) or x.shape==(self.n_x,1), f"State vector must be of shape ({self.n_x},) but got {x.shape}"
        self.x = x

    def set_control(self, u:np.ndarray):
        assert u.shape==(self.n_u,) or u.shape==(self.n_u,1), f"Control vector must be of shape ({self.n_u},) but got {u.shape}"
        self.u = u

    def dynamics(self, u:np.ndarray)-> np.ndarray:
        assert u.shape==(self.n_u,) or u.shape==(self.n_u,1), f"Control vector must be of shape ({self.n_u},)"
        self.u = u
        # Calculate the dynamics
        dx = self.fx(self.x) + self.gx(self.x)@self.u
        return dx
    
    def step(self, x=None, u=None, dt=0.1)-> np.ndarray:
        # Update the state using the dynamics
        if x is not None:
            self.set_state(x)
        if u is not None:
            self.set_control(u)
        # Calculate the dynamics
        dx = self.dynamics(u)
        self.x += dx * dt
        return self.x


class FreeFlyer(Robot):
    def __init__(self):
        super().__init__(n_x=4, n_u=2)
        # dynamics in the form: dx = f(x) + g(x)u
        self.mass = 16.8
        self.fx = lambda x: np.array([[0,0,1,0],
                                      [0,0,0,1],
                                      [0,0,0,0],
                                      [0,0,0,0]])@x
        self.gx = lambda x: np.array([[0, 0], [0, 0], [1/self.mass, 0], [0, 1/self.mass]])

        self.U = HyperRectangle(np.array([-3, -3]), np.array([3, 3]))


class BlueROV(Robot):
    def __init__(self):
        super().__init__(n_x=4, n_u=2)
        # dynamics in the form: dx = f(x) + g(x)u
        self.fx = lambda x: np.array([[0,0,1,0],
                                      [0,0,0,1],
                                      [0,0,-0.1,0],
                                      [0,0,0,-0.1]])@x
        self.gx = lambda x: np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

        self.U = HyperRectangle(np.array([-2, -2]), np.array([2, 2]))

class BlueROV2(Robot):
    def __init__(self):
        # from https://www.mdpi.com/2077-1312/10/12/1898
        super().__init__(n_x=12, n_u=6)
        # dynamics in the form: dx = f(x) + g(x)u
        # parameters:
        self.g = 9.81       # gravity, m/s^2
        self.rho = 1000     # water density, kg/m^3
        self.m = 13.5       # mass, kg 
        self.V = 0.0134     # volume, m^3
        self.I_xx = 0.26    # inertia, kg*m^2
        self.I_yy = 0.23    # inertia, kg*m^2
        self.I_zz = 0.37    # inertia, kg*m^2
        self.x_g, self.y_g, self.z_g = 0.0, 0.0, 0.0    # center of gravity, m
        self.x_b, self.y_b, self.z_b = 0.0, 0.0, -0.01  # center of buoyancy, m

        self.X_u = 13.7     # drag coefficient, N/m
        self.X_u_u = 141.0  # Ns^2/m^2
        self.Y_v = 0        # drag coefficient, N/m
        self.Y_v_v = 217.0  # Ns^2/m^2
        self.Z_w = 33.0     # drag coefficient, N/m
        self.Z_w_w = 190.0  # Ns^2/m^2
        self.K_p = 0.0      # drag coefficient, N*m/rad
        self.K_p_p = 1.19   # Ns^2/rad^2
        self.M_q = 0.8      # drag coefficient, N*m/rad
        self.M_q_q = 0.47   # Ns^2/rad^2
        self.N_r = 0.0      # drag coefficient, N*m/rad
        self.N_r_r = 1.5    # Ns^2/rad^2

        self.X_du = 6.36    # kg
        self.Y_dv = 7.12    # kg
        self.Z_dw = 18.68   # kg
        self.K_dp = 0.189   # kg*m^2
        self.M_dq = 0.135   # kg*m^2
        self.N_dr = 0.222   # kg*m^2

        self.fx = lambda x: self._fx(x)
        self.gx = lambda x: self._gx(x)

        self.U = HyperRectangle(np.array([-85, -85, -120, -26, -14, -22]),
                                np.array([85, 85, 120, 26, 14, 22]))
        
    def _fx(self,x):
        eta, v = x[0:6], x[6:12]
        M_inv = cs.DM(np.linalg.inv(self.M(x)))
        fx = cs.blockcat([[(self.J(x))@v],
                          [M_inv@(-self.C(x)@v - self.D(x)@v - self.g_(x))]])
        return fx
    
    def _gx(self,x):
        M_inv = cs.DM(np.linalg.inv(self.M(x)))
        gx = cs.blockcat([[cs.GenDM_zeros((6,6))],
                          [M_inv]])
        return gx

    def J(self,x):
        phi, theta, psi = x[3], x[4], x[5]
        cos = cs.cos
        sin = cs.sin
        tan = cs.tan
        J_1 = cs.blockcat([[cos(psi)*cos(theta), -sin(psi)*cos(phi)+cos(psi)*sin(theta)*sin(phi), sin(psi)*sin(phi)+cos(psi)*cos(phi)*sin(theta)],
                        [sin(psi)*cos(theta), cos(psi)*cos(phi)+sin(phi)*sin(theta)*sin(psi), -cos(psi)*sin(phi)+sin(theta)*sin(psi)*cos(phi)],
                        [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)]])
        J_2 = cs.blockcat([[1., sin(phi)*tan(theta), cos(phi)*tan(theta)],
                           [0, cos(phi), -sin(phi)],
                           [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]])
        
        J = cs.blockcat([[J_1, cs.GenDM_zeros((3,3))],
                         [cs.GenDM_zeros((3,3)), J_2]])
        return J
    
    def M(self,x):
        M_RB = cs.blockcat([[self.m*cs.diag([1,1,1]), cs.GenDM_zeros((3,3))],
                         [cs.GenDM_zeros((3,3)), cs.diag([self.I_xx, self.I_yy, self.I_zz])]])
        M_A = -np.diag([self.X_du, self.Y_dv, self.Z_dw, self.K_dp, self.M_dq, self.N_dr])
        M = M_RB + M_A
        return M
    
    def C(self,x):
        u, v, w, p, q, r = x[6], x[7], x[8], x[9], x[10], x[11]
        m = self.m
        C_RB = cs.blockcat([[0, 0, 0, 0, m*w, -m*v],
                         [0, 0, 0, -m*w, 0, m*u],
                         [0, 0, 0, m*v, -m*u, 0],
                         [0, m*w, -m*v, 0, -self.I_zz*r, -self.I_yy*q],
                         [-m*w, 0, m*u, self.I_zz*r, 0, self.I_xx*p],
                         [m*v, -m*u, 0, self.I_yy*q, -self.I_xx*p, 0]])
        C_A = cs.blockcat([[0, 0, 0, 0, -self.Z_dw*w, self.Y_dv*v],
                        [0, 0, 0, self.Z_dw*w, 0, -self.X_du*u],
                        [0, 0, 0, -self.Y_dv*v, self.X_du*u, 0],
                        [0, -self.Z_dw*w, self.Y_dv*v, 0, -self.N_dr*r, self.M_dq*q],
                        [self.Z_dw*w, 0, -self.X_du*u, self.N_dr*r, 0, -self.K_dp*p],
                        [-self.Y_dv*v, self.X_du*u, 0, -self.M_dq*q, self.K_dp*p, 0]])
        C = C_RB + C_A
        return C
    
    def D(self,x):
        u, v, w, p, q, r = x[6], x[7], x[8], x[9], x[10], x[11]
        abs = lambda x: cs.sqrt(x**2 + 1e-6)
        D = -cs.blockcat([[self.X_u, 0, 0, 0, 0, 0],
                          [0, self.Y_v, 0, 0, 0, 0],
                          [0, 0, self.Z_w, 0, 0, 0],
                          [0, 0, 0, self.K_p, 0, 0],
                          [0, 0, 0, 0, self.M_q, 0],
                          [0, 0, 0, 0, 0, self.N_r]])
        D_n = -cs.blockcat([[self.X_u_u*abs(u), 0, 0, 0, 0, 0],
                        [0, self.Y_v_v*abs(v), 0, 0, 0, 0],
                        [0, 0, self.Z_w_w*abs(w), 0, 0, 0],
                        [0, 0, 0, self.K_p_p*abs(p), 0, 0],
                        [0, 0, 0, 0, self.M_q_q*abs(q), 0],
                        [0, 0, 0, 0, 0, self.N_r_r*abs(r)]])
        D = D + D_n
        return D
    
    def g_(self,x):
        W = self.m*self.g
        B = self.rho*self.g*self.V
        phi, theta, psi = x[3], x[4], x[5]
        sin = cs.sin
        cos = cs.cos

        g = cs.blockcat([[(W-B)*sin(theta)],
                      [-(W-B)*cos(theta)*sin(phi)],
                      [-(W-B)*cos(theta)*cos(phi)],
                      [self.y_b*B*cos(theta)*cos(phi) - self.z_b*B*cos(theta)*sin(phi)],
                      [-self.z_b*B*sin(theta) - self.x_b*B*cos(theta)*cos(phi)],
                      [self.x_b*B*cos(theta)*sin(phi) - self.y_b*B*sin(theta)]])
        return g
    
