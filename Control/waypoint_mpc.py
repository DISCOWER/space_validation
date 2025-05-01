import numpy as np
import casadi as cs
import time
import matplotlib.pyplot as plt

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from Utilities.Robots import FreeFlyer, BlueROV, BlueROV2

# simple waypoint mpc, randomizes the initial state
# and and controls the robot towards [0,0,0,0]

class WaypointMPC:
    def __init__(self, robot:BlueROV2, N=10, dt=0.1):
        self.robot = robot
        self.N = N
        self.dt = dt
        self.x0 = np.zeros((self.robot.n_x,))

        self.Q = np.diag([5e0, 5e0, 5e0, 2e-1, 2e-1, 2e-1, 
                          5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2])
        self.Q_e = 10*self.Q
        self.R = np.diag([5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4])

        self.params = {}
        self.vars = {}

        self.ocp = self.setup()

    def setup(self):
        ocp = cs.Opti()
        X = ocp.variable(self.robot.n_x, self.N)
        U = ocp.variable(self.robot.n_u, self.N)

        x0 = ocp.parameter(self.robot.n_x)
        xdes = ocp.parameter(self.robot.n_x)

        # set initial state
        ocp.subject_to(X[:,0] == x0)

        # set dynamics constraints
        for i in range(self.N-1):
            ocp.subject_to(X[:,i+1] == self.robot.step(X[:,i], U[:,i], self.dt))
        
        # control input constraints
        for i in range(self.N):
            ocp.subject_to(self.robot.U.lower_bounds <= U[:,i])
            ocp.subject_to(U[:,i] <= self.robot.U.upper_bounds)

        # cost function
        cost_eq = 0
        for i in range(self.N):
            cost_eq += (X[:,i] - xdes).T@self.Q@(X[:,i] - xdes)
            cost_eq += (U[:,i]).T@self.R@(U[:,i])
        cost_eq += (X[:,-1] - xdes).T@self.Q_e@(X[:,-1] - xdes)
        
        ocp.minimize(cost_eq)

        # solver method
        opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.sb': 'yes',
                'verbose':False}
        ocp.solver('ipopt',opts)

        self.params['x0'] = x0
        self.params['xdes'] = xdes

        self.vars['X'] = X
        self.vars['U'] = U
        
        return ocp
    
    def solve(self, x0, xdes,
              initial_guess={'X': None, 'U': None}):
        t0 = time.time()

        if initial_guess['X'] is not None:
            self.ocp.set_initial(self.vars['X'], initial_guess['X'])
        if initial_guess['U'] is not None:
            self.ocp.set_initial(self.vars['U'], initial_guess['U'])

        self.ocp.set_value(self.params['x0'], x0)
        self.ocp.set_value(self.params['xdes'], xdes)

        try:
            sol = self.ocp.solve()
            X_pred = sol.value(self.vars['X'])
            U_pred = sol.value(self.vars['U'])
            print(f"Solution found in {time.time()-t0:.2f} seconds")
        except Exception as e:
            print(f"Error: {e}")
            X_pred = np.zeros((self.N, self.robot.n_x))
            U_pred = np.zeros((self.N, self.robot.n_u))

        return X_pred, U_pred


if __name__ == "__main__":
    robot = BlueROV2()
    mpc = WaypointMPC(robot, N=10, dt=0.1)

    x0 = np.array([0, 0, 1, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0])
    xdes = np.array([1, 1, 0, 0, 0, np.pi/2,
                     0, 0, 0, 0, 0, 0])

    Nsim = 30
    x = x0
    x_hist = np.zeros((robot.n_x, Nsim))
    for i in range(Nsim):
        X_pred, U_pred = mpc.solve(x, xdes)
        # print(f"X_pred: {X_pred}")
        x = X_pred[:,1]
        x_hist[:,i] = x
        

    # 2D top down plot of the trajectory
    fig, ax = plt.subplots()
    ax.plot(x_hist[0,:], x_hist[1,:], 'r-')
    ax.plot(x0[0], x0[1], 'go')
    ax.plot(xdes[0], xdes[1], 'bo')

    fig.savefig("figures/waypoint_mpc_2d.png")

    # 3D plot of an arrow pointing in the right direction
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x_hist[0,:], x_hist[1,:], x_hist[2,:], 
              np.cos(x_hist[5,:]), np.sin(x_hist[5,:]), 0, 
              length=0.1, normalize=True)
    ax.set_title('3D trajectory')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    ax.view_init(elev=20, azim=30)
    fig.savefig("figures/waypoint_mpc_3d.png")
    
