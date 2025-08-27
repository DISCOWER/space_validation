import numpy as np
import casadi as cs
import gurobipy as gp
from Utilities.sets import HyperRectangle, Zonotope
from Utilities.rotations import skew_symmetric_cs, skew_symmetric_np, q_to_rot_mat_cs, q_to_rot_mat_np
from Utilities.rotations import euler_to_quat_cs, euler_to_quat_np
from Utilities.rotations import quat_mult
from Utilities.sets import compute_K
from Utilities.stl import OptProbItems


class Robot:
    def __init__(self,n_x:int,n_u:int):
        self.n_x = n_x  # Number of state variables
        self.n_u = n_u  # Number of control inputs
        self.x = np.zeros((n_x,))
        self.u = np.zeros((n_u,))

        self.fx = None  # State dynamics function
        self.gx = None

    def calculate_fx(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def calculate_gx(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def set_state(self, x:np.ndarray):
        # assert x.shape==(self.n_x,) or x.shape==(self.n_x,1), f"State vector must be of shape ({self.n_x},) but got {x.shape}"
        self.x = x

    def set_control(self, u:np.ndarray):
        # assert u.shape==(self.n_u,) or u.shape==(self.n_u,1), f"Control vector must be of shape ({self.n_u},) but got {u.shape}"
        self.u = u

    def dynamics(self, u:np.ndarray)-> np.ndarray:
        # assert u.shape==(self.n_u,) or u.shape==(self.n_u,1), f"Control vector must be of shape ({self.n_u},)"
        self.u = u
        # Calculate the dynamics
        dx = self.calculate_fx(self.x) + self.calculate_gx(self.x)@self.u
        return dx
    
    def step(self, x=None, u=None, dt=0.1)-> np.ndarray:
        # # Update the state using the dynamics
        # if x is not None:
        #     self.set_state(x)
        # if u is not None:
        #     self.set_control(u)
        # # Calculate the dynamics
        # dx = self.dynamics(u)
        # self.x += dx * dt
        return x + (self.calculate_fx(x) + self.calculate_gx(x)@u) * dt
        # return self.x

    def add_state_constraints(self, prog:gp.Model, items:OptProbItems):
        # Add constraitns on the state that are fundamental:
        # - rotations in [0,2pi]
        # - with additional wrapping? (TODO)
        pass


class LinearFreeFlyer2DoF(Robot):
    def __init__(self):
        super().__init__(n_x=4, n_u=2)
        # dynamics in the form: dx = f(x) + g(x)u
        self.mass = 16.8

        self.A = np.array([[0,0,1,0],
                           [0,0,0,1],
                           [0,0,0,0],
                           [0,0,0,0]])
        self.B = np.array([[0, 0],
                           [0, 0],
                           [1/self.mass, 0],
                           [0, 1/self.mass]])
        self.C = np.array([[0, 0],
                           [0, 0],
                           [1/20, 0],
                           [0, 1/20]])
        self.K = compute_K(self.B, self.C)

        self.U = HyperRectangle(np.array([-3, -3]), np.array([3, 3]))
        self.D = HyperRectangle(np.array([-1, -1]), np.array([1, 1]))

        # self.U_effective = minkowski_difference(self.U, self.K@self.D)
        self.KD = HyperRectangle(
            self.K[2:,2:]@self.D.lower_bounds,
            self.K[2:,2:]@self.D.upper_bounds
        )
        self.U_effective = self.U.subtract(self.KD)

    def calculate_fx(self, x):
        return self.A@x
    def calculate_gx(self, x):
        return self.B

class LinearFreeFlyer6DoF(Robot):
    def __init__(self):
        super().__init__(n_x=12, n_u=6)
        # dynamics in the form: dx = f(x) + g(x)u
        self.mass = 16.8 #(16.8: empty, 17.8: full)  # kg
        self.inertia = np.diag((0.1454, 0.1366, 0.1594))

        self.A = np.zeros((12, 12))
        self.A[0:6, 6:12] = np.eye(6)

        self.B = np.zeros((12, 6))
        self.B[6:12, :] = np.array([[1/self.mass, 0, 0, 0, 0, 0],
                                    [0, 1/self.mass, 0, 0, 0, 0],
                                    [0, 0, 1/self.mass, 0, 0, 0],
                                    [0, 0, 0, 1/self.inertia[0,0], 0, 0],
                                    [0, 0, 0, 0, 1/self.inertia[1,1], 0],
                                    [0, 0, 0, 0, 0, 1/self.inertia[2,2]]])
        
        self.C = np.zeros((12, 3))
        # TODO: deal with positive and negative values here
        self.C[6:9, :] = np.array([[1/20, 0, 0],
                                   [0, 1/20, 0],
                                   [0, 0, 1/20]])
        self.K = np.linalg.pinv(self.B)@self.C #compute_K(self.B, self.C)

        # Define the control input bounds
        max_thrust = 2.125
        max_torque = 0.714
        scale_thrust = 2/3
        scale_torque = 1/3
        self.U = HyperRectangle(
            np.array([-scale_thrust*max_thrust]*3 + [-scale_torque*max_torque]*3),
            np.array([scale_thrust*max_thrust]*3 + [scale_torque*max_torque]*3)
        )
        max_floor_force = (36*self.mass)/1000
        self.D = HyperRectangle(np.array(3*[-max_floor_force]),np.array(3*[max_floor_force]))

        # self.U_effective = minkowski_difference(self.U, self.K@self.D)
        # TODO: deal with the fact that K and D are not of same dimension
        self.KD = HyperRectangle(
            self.K@self.D.lower_bounds,
            self.K@self.D.upper_bounds
        )
        self.U_effective = self.U.subtract(self.KD)
        print(f"U_effective: {self.U_effective.center}, {self.U_effective.lower_bounds}, {self.U_effective.upper_bounds}")

    def calculate_fx(self,x):
        return self.A@x
    def calculate_gx(self,x):
        return self.B

    def add_state_constraints(self, prog:gp.Model, items:OptProbItems):
        for i in range(items.x_vars.shape[0]):
            # # Add constraints for roll and yaw to be between [-pi, pi]
            # prog.addConstr(items.x_vars[i, 4] >= -np.pi, f"roll_lower_{i}")
            # prog.addConstr(items.x_vars[i, 4] <= np.pi, f"roll_upper_{i}")
            # prog.addConstr(items.x_vars[i, 5] >= -np.pi, f"yaw_lower_{i}")
            # prog.addConstr(items.x_vars[i, 5] <= np.pi, f"yaw_upper_{i}")
            # # Add constrains for pitch to be between [-pi/2, pi/2] (prevent gymbal lock)
            # prog.addConstr(items.x_vars[i, 3] >= -np.pi/2, f"pitch_lower_{i}")
            # prog.addConstr(items.x_vars[i, 3] <= np.pi/2, f"pitch_upper_{i}")
            # Add constraints for roll to be 0
            # TODO: make this a user-defined dimension (based on order of euler angles)
            prog.addConstr(items.x_vars[i, 4] >= -np.pi, f"roll_lower_{i}")
            prog.addConstr(items.x_vars[i, 4] <= np.pi, f"roll_upper_{i}")


            # Add constraints for the angular velocities to be between [-pi/8, pi/8]
            prog.addConstr(items.x_vars[i, 9] >= -np.pi/8, f"p_lower_{i}")
            prog.addConstr(items.x_vars[i, 9] <= np.pi/8, f"p_upper_{i}")
            prog.addConstr(items.x_vars[i, 10] >= -np.pi/8, f"q_lower_{i}") 
            prog.addConstr(items.x_vars[i, 10] <= np.pi/8, f"q_upper_{i}")
            prog.addConstr(items.x_vars[i, 11] >= -np.pi/8, f"r_lower_{i}")
            prog.addConstr(items.x_vars[i, 11] <= np.pi/8, f"r_upper_{i}")

class FreeFlyer(Robot):
    def __init__(self,
                 dt=0.02,
                 iX:cs.SX|cs.MX=cs.SX):
        self.dt = dt
        self.iX = iX
        # state: x = [p, q, v, w]
        # where q is a scalar first unit quaternion
        nx = 13
        nu = 6
        super().__init__(n_x=nx, n_u=nu)

        # dynamics in the form: dx = f(x) + g(x)u
        self.mass = 16.8 #(16.8: empty, 17.8: full)  # kg
        self.inertia = np.diag((0.1454, 0.1366, 0.1594))
        max_thrust = 2.125
        max_torque = 0.714
        scale_thrust = 2/3
        scale_torque = 1/3
        self.U = HyperRectangle(
            np.array([-scale_thrust*max_thrust]*3 + [-scale_torque*max_torque]*3),
            np.array([scale_thrust*max_thrust]*3 + [scale_torque*max_torque]*3)
        )

        self.create_fx()
        self.create_gx()
        self.create_cx()

        # TODO: deal with the fact that K and D are not of same dimension
        max_floor_force = (36*self.mass)/1000
        self.D = HyperRectangle(
            np.array(3*[-max_floor_force]),
            np.array(3*[max_floor_force])
        )
        self.create_K()
        self.create_U_effective()
        # U_effective = self.calculate_U_effective(self.x)
        # print(f"U_effective: {U_effective.center[:,0].T}, {U_effective.lower_bounds[:,0].T}, {U_effective.upper_bounds[:,0].T}")

        
    def create_K(self):
        if not hasattr(self, '_K_sym'):
            p = self.iX.sym('p', 3)
            q = self.iX.sym('q', 4)
            v = self.iX.sym('v', 3)
            w = self.iX.sym('w', 3)
            self._K_sym = cs.Function('K', [p, q, v, w],
                [
                    cs.pinv(self.calculate_gx(cs.vertcat(p, q, v, w))) @ self.calculate_cx(cs.vertcat(p, q, v, w))
                ]
            )
    def calculate_K(self, x):
        p, q, v, w = x[0:3], x[3:7], x[7:10], x[10:13]
        if isinstance(p, np.ndarray):
            return np.array(self._K_sym(p, q, v, w))
        else:
            return self._K_sym(p, q, v, w)


    def create_U_effective(self):
        if not hasattr(self, '_U_effective_sym'):
            p = self.iX.sym('p', 3)
            q = self.iX.sym('q', 4)
            v = self.iX.sym('v', 3)
            w = self.iX.sym('w', 3)
            self._U_effective_sym = cs.Function('U_effective', [p, q, v, w],
                [
                    self.U.lower_bounds - self._K_sym(p, q, v, w)@self.D.lower_bounds,
                    self.U.upper_bounds - self._K_sym(p, q, v, w)@self.D.upper_bounds
                ]
            )
    def calculate_U_effective(self, x):
        p, q, v, w = x[0:3], x[3:7], x[7:10], x[10:13]
        lb, ub = self._U_effective_sym(p, q, v, w)
        if isinstance(x, np.ndarray):
            return HyperRectangle(np.array(lb), np.array(ub))
        else:
            return HyperRectangle(lb, ub)
        

    def create_dynamics(self):
        if not hasattr(self, '_dynamics_sym'):
            p = self.iX.sym('p', 3)
            q = self.iX.sym('q', 4)
            v = self.iX.sym('v', 3)
            w = self.iX.sym('w', 3)
            u = self.iX.sym('u', 6)
            self._dynamics_sym = cs.Function('dynamics', [p, q, v, w, u],
                [
                    self.calculate_fx(cs.vertcat(p, q, v, w)) + self.calculate_gx(cs.vertcat(p, q, v, w)) @ u
                ]
            )
    def calculate_dynamics(self, x, u):
        p, q, v, w = x[0:3], x[3:7], x[7:10], x[10:13]
        if not hasattr(self, '_dynamics_sym'):
            self.create_dynamics()
        if isinstance(x, np.ndarray):
            return np.array(self._dynamics_sym(p,q,v,w,u)).reshape((self.n_x,))
        else:
            return self._dynamics_sym(p,q,v,w,u)


    def create_fx(self):
        if not hasattr(self, '_fx_sym'):
            p = self.iX.sym('p', 3)
            q = self.iX.sym('q', 4)
            v = self.iX.sym('v', 3)
            w = self.iX.sym('w', 3)
            w_cross = cs.vertcat(
                cs.horzcat(0, -w[2], w[1]),
                cs.horzcat(w[2], 0, -w[0]),
                cs.horzcat(-w[1], w[0], 0)
            )
            self._fx_sym = cs.Function('fx', [p, q, v, w],
                [
                    cs.blockcat([
                        [v],
                        # [q_to_rot_mat_cs(q) @ v],
                        [0.5 * quat_mult(q, cs.vertcat(0, w))],
                        [self.iX.zeros(3,)],
                        # [self.iX.zeros(3,)]
                        [-self.iX(np.linalg.inv(self.inertia)) @ cs.mtimes(w_cross, cs.mtimes(self.inertia, w))]
                    ])
                ]
            )
    def calculate_fx(self, x):
        p, q, v, w = x[0:3], x[3:7], x[7:10], x[10:13]
        # if p is a numpy array
        if isinstance(p, np.ndarray):
            return np.array(self._fx_sym(p, q, v, w)).reshape((self.n_x,))
        else:
            return self._fx_sym(p, q, v, w)
    

    def create_gx(self):
        if not hasattr(self, '_gx_sym'):
            p = self.iX.sym('p', 3)
            q = self.iX.sym('q', 4)
            v = self.iX.sym('v', 3)
            w = self.iX.sym('w', 3)
            self._gx_sym = cs.Function('gx', [p, q, v, w],
                [
                    cs.blockcat([
                        [self.iX.zeros((3, 6))],
                        [self.iX.zeros((4, 6))],
                        [cs.horzcat(
                            # self.iX.eye(3) / self.mass,
                            q_to_rot_mat_cs(q) / self.mass,
                            self.iX.zeros((3, 3))
                        )],
                        [cs.horzcat(
                            self.iX.zeros((3, 3)),
                            self.iX(np.linalg.inv(self.inertia))
                        )]
                    ])
                ]
            )
    def calculate_gx(self, x):
        p, q, v, w = x[0:3], x[3:7], x[7:10], x[10:13]
        if isinstance(p, np.ndarray):
            return np.array(self._gx_sym(p, q, v, w))
        else:
            return self._gx_sym(p, q, v, w)
        

    def create_cx(self):
        if not hasattr(self, '_cx_sym'):
            p = self.iX.sym('p', 3)
            q = self.iX.sym('q', 4)
            v = self.iX.sym('v', 3)
            w = self.iX.sym('w', 3)
            cx = self.iX.zeros((13,3))
            cx[7:10, 0:3] = self.iX.eye(3)/self.mass
            self._cx_sym = cs.Function('cx', [p, q, v, w],
                [
                    cx
                ]
            )
    def calculate_cx(self, x):
        p, q, v, w = x[0:3], x[3:7], x[7:10], x[10:13]
        if isinstance(p, np.ndarray):
            return np.array(self._cx_sym(p, q, v, w))
        else:
            return self._cx_sym(p, q, v, w)