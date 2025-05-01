import numpy as np
import matplotlib.pyplot as plt
import copy

# reach-avoid gurobi optimization problem
# might want to add integer variables later so we can use gurobi to solve the problem
import gurobipy as gp

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from Utilities.Robots import FreeFlyer, BlueROV, BlueROV2
from Utilities.helpers import HyperRectangle

# hyperparameters
N = 100     # number of time steps
dt = 0.1    # time step size
bigM = 1e4

# specification
X0 = HyperRectangle(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
Xf = HyperRectangle(np.array([0.95, 0.95, 0, 0]), np.array([1, 1, 0, 0]))

Obs = [HyperRectangle(np.array([0.4, -0.1, 0, 0]), np.array([0.6, 0.6, 0, 0])),
       HyperRectangle(np.array([0.75, 0.5, 0, 0]), np.array([0.9, 1.1, 0, 0]))]

World = HyperRectangle(np.array([0, 0, -bigM, -bigM]), np.array([1, 1, bigM, bigM]))
# robot
robot = FreeFlyer()

# create planner
if False:
    opt = gp.Model("prob1")
    x_vars = opt.addMVar((N, robot.n_x), lb=-np.inf, ub=np.inf, name="x")
    u_vars = opt.addMVar((N, robot.n_u), lb=-np.inf, ub=np.inf, name="u")
    alpha_vars = opt.addVar(lb=0, ub=1, name="alpha")

    for i in range(N):
        opt.addConstrs((u_vars[i, j] >= alpha_vars*robot.U.lower_bounds[j] for j in range(2)))
        opt.addConstrs((u_vars[i, j] <= alpha_vars*robot.U.upper_bounds[j] for j in range(2)))

    # constraints
    opt.addConstrs((x_vars[i+1,:] == robot.step(x_vars[i, :], u_vars[i, :], dt) for i in range(N-1)))
    opt.addConstrs((x_vars[i, :] >= World.lower_bounds for i in range(N)))
    opt.addConstrs((x_vars[i, :] <= World.upper_bounds for i in range(N)))

    opt.addConstr(x_vars[0, :] >= X0.lower_bounds)
    opt.addConstr(x_vars[0, :] <= X0.upper_bounds)
    opt.addConstr(x_vars[-1, :] >= Xf.lower_bounds)
    opt.addConstr(x_vars[-1, :] <= Xf.upper_bounds)

    for obs in Obs:
        z_vars = opt.addMVar((N, 4), vtype=gp.GRB.BINARY, name="z")
        for i in range(N):
            # x_vars[i,0:2] should be outside one of the faces of the obstacle
            opt.addConstr(x_vars[i,0] <= obs.lower_bounds[0] + bigM*(1-z_vars[i,0]))
            opt.addConstr(x_vars[i,0] >= obs.upper_bounds[0] - bigM*(1-z_vars[i,1]))
            opt.addConstr(x_vars[i,1] <= obs.lower_bounds[1] + bigM*(1-z_vars[i,2]))
            opt.addConstr(x_vars[i,1] >= obs.upper_bounds[1] - bigM*(1-z_vars[i,3]))
            opt.addConstr(gp.quicksum(z_vars[i,:]) >= 1)

    # objective
    opt.setObjective(alpha_vars, gp.GRB.MINIMIZE)
    opt.setParam('OutputFlag', 0)  # Suppress Gurobi output
    opt.optimize()
    if opt.status == gp.GRB.OPTIMAL:
        print(f"Optimal solution found")
        print(f"Alpha: {alpha_vars.X}")
    else:
        print(f"No optimal solution found")
        print(f"Status: {opt.status}")

    x_ff = x_vars.X
    u_ff = u_vars.X
    alpha = alpha_vars.X
    save_vars = {'x_ff': x_ff, 'u_ff': u_ff, 'alpha': alpha}
    np.savez("solutions/ff_solution.npz", **save_vars)
else:
    # load the solution
    data = np.load("solutions/ff_solution.npz")
    x_ff = data['x_ff']
    u_ff = data['u_ff']
    alpha = data['alpha']

# plot the trajectory
fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].plot(x_ff[:, 0], x_ff[:, 1], 'g-')
X0.plot(axs[0], color='green', alpha=0.5)
Xf.plot(axs[0], color='blue', alpha=0.5)
[obs.plot(axs[0], color='red', alpha=0.5) for obs in Obs]
axs[0].set_aspect('equal', adjustable='box')

axs[1].plot(u_ff[:,0])
axs[1].plot(u_ff[:,1])
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Control input')
plt.savefig("figures/sp_trajectory.png")

# feedback linearization controller for the underwater robot to behave like a free flyer
uw_robot = BlueROV2()

# u_fbl = lambda x, v: np.linalg.pinv(uw_robot.gx(x))@(robot.fx(x) + robot.gx(x)@v - uw_robot.fx(x))
#! bluerov2
def u_fbl(x, v):
    # add 4 zeros behind x[0:2] and 4 zeros behind x[2:4]
    x_uw = np.zeros((uw_robot.n_x,))
    x_uw[0:2] = x[0:2]
    x_uw[6:8] = x[2:4]
    # add 4 zeros behind v[0:2]
    v_uw = np.zeros((uw_robot.n_u,))
    v_uw[0:2] = v[0:2]
    # compute the control input
    u_fbl = np.linalg.pinv(uw_robot.gx(x_uw)[[0,1,6,7],0:2])@(robot.fx(x) + robot.gx(x)@v - uw_robot.fx(x_uw)[[0,1,6,7]])
    u_fbl = np.concatenate((u_fbl, np.zeros((4,1))))
    return u_fbl

x = copy.deepcopy(x_ff[0, :])
x_hist = np.zeros((N, uw_robot.n_x))
u_hist = np.zeros((N, uw_robot.n_u))
for i in range(N):
    u = u_fbl(x, u_ff[i, :])
    x = uw_robot.step(np.concatenate((x[0:2],np.zeros((4,)),x[2:4],np.zeros((4,)))), u, dt)
    x = np.array(x).squeeze()
    u = np.array(u).squeeze()
    x_hist[i, :] = x
    u_hist[i, :] = u
    x = np.concatenate((x[0:2],x[6:8]))

fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].plot(x_hist[:, 0], x_hist[:, 1], 'g-')
X0.plot(axs[0], color='green', alpha=0.5)
Xf.plot(axs[0], color='blue', alpha=0.5)
[obs.plot(axs[0], color='red', alpha=0.5) for obs in Obs]
axs[0].set_aspect('equal', adjustable='box')
axs[1].plot(u_hist[:,0])
axs[1].plot(u_hist[:,1])
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Control input')
plt.savefig("figures/sp_trajectory_uw.png")


u_max = np.max(np.abs(u_hist), axis=0)
print(f"Alpha for uw: {np.max(u_max/uw_robot.U.upper_bounds)}")

# So now we have to find \phi (for now deltaT) such that
# alpha for uw is equal to alpha for the free flyer
# we have bilinear constraints so that is quite tricky but
# we have good initial guesses regarding obstacle avoidance
if False:
    opt = gp.Model("prob2")
    x_vars = opt.addMVar((N, uw_robot.n_x), lb=-np.inf, ub=np.inf, name="x_uw")
    u_vars = opt.addMVar((N, uw_robot.n_u), lb=-np.inf, ub=np.inf, name="u_uw")
    dt_vars = opt.addVar(lb=0, ub=1, name="dt_uw")
    opt.update()

    x_vars.Start = x_ff
    u_vars.Start = u_ff
    dt_vars.Start = dt

    for i in range(N):
        opt.addConstrs((u_vars[i, j] >= alpha*(robot.U.lower_bounds[j]-u_fbl(x_vars[i,:],u_vars[i,:])[j]) for j in range(2)))
        opt.addConstrs((u_vars[i, j] <= alpha*(robot.U.upper_bounds[j]-u_fbl(x_vars[i,:],u_vars[i,:])[j]) for j in range(2)))

    # constraints
    opt.addConstrs((x_vars[i+1,:] == uw_robot.step(x_vars[i, :], u_vars[i, :], dt_vars) for i in range(N-1)))
    opt.addConstrs((x_vars[i, :] >= World.lower_bounds for i in range(N)))
    opt.addConstrs((x_vars[i, :] <= World.upper_bounds for i in range(N)))

    opt.addConstr(x_vars[0, :] >= X0.lower_bounds)
    opt.addConstr(x_vars[0, :] <= X0.upper_bounds)
    opt.addConstr(x_vars[-1, :] >= Xf.lower_bounds)
    opt.addConstr(x_vars[-1, :] <= Xf.upper_bounds)

    for obs in Obs:
        z_vars = opt.addMVar((N, 4), vtype=gp.GRB.BINARY, name="z")
        for i in range(N):
            # x_vars[i,0:2] should be outside one of the faces of the obstacle
            opt.addConstr(x_vars[i,0] <= obs.lower_bounds[0] + bigM*(1-z_vars[i,0]))
            opt.addConstr(x_vars[i,0] >= obs.upper_bounds[0] - bigM*(1-z_vars[i,1]))
            opt.addConstr(x_vars[i,1] <= obs.lower_bounds[1] + bigM*(1-z_vars[i,2]))
            opt.addConstr(x_vars[i,1] >= obs.upper_bounds[1] - bigM*(1-z_vars[i,3]))
            opt.addConstr(gp.quicksum(z_vars[i,:]) >= 1)

    # objective
    time_diff_var = opt.addVar(lb=-np.inf, ub=np.inf, name="time_diff")
    opt.addConstr(time_diff_var == N*dt_vars - N*dt)
    cost_var = opt.addVar(lb=-np.inf, ub=np.inf, name="cost")
    opt.addConstr(cost_var == gp.abs_(time_diff_var))
    opt.setObjective(cost_var, gp.GRB.MINIMIZE)
    opt.setParam('OutputFlag', 1)  # Suppress Gurobi output
    opt.optimize()
    if opt.status == gp.GRB.OPTIMAL:
        print(f"Optimal solution found")
        print(f"dt: {dt_vars.X}")
    else:
        print(f"No optimal solution found")
        print(f"Status: {opt.status}")

    x_uw = x_vars.X
    u_uw = u_vars.X
    dt_uw = dt_vars.X
    save_vars = {'x_uw': x_uw, 'u_uw': u_uw, 'dt_uw': dt_uw}
    np.savez("solutions/uw_solution.npz", **save_vars)
else:
    # load the solution
    data = np.load("solutions/uw_solution.npz")
    x_uw = data['x_uw']
    u_uw = data['u_uw']
    dt_uw = data['dt_uw']

# plot the trajectory
fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].plot(x_ff[:, 0], x_ff[:, 1], 'g-')
axs[0].plot(x_uw[:, 0], x_uw[:, 1], 'c-')
X0.plot(axs[0], color='green', alpha=0.5)
Xf.plot(axs[0], color='blue', alpha=0.5)
[obs.plot(axs[0], color='red', alpha=0.5) for obs in Obs]
axs[0].set_aspect('equal', adjustable='box')

axs[1].plot(u_uw[:,0])
axs[1].plot(u_uw[:,1])
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Control input')
plt.savefig("figures/sp_uw_trajectory.png")
