import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation as R

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from Utilities.beziers import eval_bezier
from Utilities.Robots import FreeFlyer, LinearFreeFlyer6DoF
from space_validation.Utilities.plotting.plotting import plot_planning_results

data = np.load('space_validation/Planning/solutions/sp_solution_nl.npz')

ff_lin = LinearFreeFlyer6DoF()
x_lin = data['x_ff']
u_lin = data['u_ff']
times = data['times']
dt = data['dt']

# FreeFlyer is 6dof nonlinear but we plan for a linear system
# in 5dof (position, orientation with roll=0). Is the resulting
# trajectory dynamically feasible? is it of equal degree of difficulty?
ff = FreeFlyer()

x_nl = np.zeros_like(x_lin)
x_nl[0,:] = x_lin[0,:]
for i in range(u_lin.shape[0]):
    # convert u from ENU to body FLU body frame
    rot = R.from_quat(x_nl[i,3:7],scalar_first=True)
    # u = np.hstack((rot.inv().apply(u_lin[i,0:3]), u_lin[i,3:6]))#rot.inv().apply(u_lin[i,3:6])))
    u = u_lin[i,:]

    x_nl[i+1,:] = ff.step(x_nl[i,:], u, dt)
    # re-normalize the quaternion
    x_nl[i+1,3:7] /= np.linalg.norm(x_nl[i+1,3:7])

print(f"x_nl[0,:]: {x_nl[0,:]}")
print(f"x_lin[0,:]: {x_lin[0,:]}")

# Plot the results
fig, axs = plt.subplots(1,4, figsize=(15,5))
axs = plot_planning_results(ff_lin, times, x_lin, u_lin,
                      in_axs=None, plot=True,
                      path="space_validation/Planning/figures/test_flatness_lin.png")
fig, axs = plt.subplots(1,4, figsize=(15,5))
plot_planning_results(ff, times, x_nl, u_lin,
                      in_axs=None, plot=True,
                      path="space_validation/Planning/figures/test_flatness_nl.png")

