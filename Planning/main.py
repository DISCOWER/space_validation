import numpy as np
import matplotlib.pyplot as plt
import copy

# reach-avoid gurobi optimization problem
# might want to add integer variables later so we can use gurobi to solve the problem
import gurobipy as gp
import casadi as cs

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
X0 = HyperRectangle(np.array([0, 0, 0, 0]), np.array([0.1, 0.1, 0, 0]))
Xf = HyperRectangle(np.array([0.95, 0.95, 0, 0]), np.array([1, 1, 0, 0]))
XA = HyperRectangle(np.array([0.2, 0.8, -0.1, -0.1]), np.array([0.4, 1.0, 0.1, 0.1]))

Obs = [HyperRectangle(np.array([0.4, -0.1, 0, 0]), np.array([0.6, 0.6, 0, 0])),
       HyperRectangle(np.array([0.75, 0.5, 0, 0]), np.array([0.9, 1.1, 0, 0]))]

World = HyperRectangle(np.array([0, 0, -bigM, -bigM]), np.array([1, 1, bigM, bigM]))
# robot
sp_robot = FreeFlyer()

# create planner
if True:
    opt = gp.Model("prob1")
    x_vars = opt.addMVar((N, sp_robot.n_x), lb=-np.inf, ub=np.inf, name="x")
    u_vars = opt.addMVar((N, sp_robot.n_u), lb=-np.inf, ub=np.inf, name="u")
    alpha_vars = opt.addVar(lb=0, ub=1, name="alpha")

    for i in range(N):
        opt.addConstrs((u_vars[i, j] >= alpha_vars*sp_robot.U.lower_bounds[j] for j in range(2)))
        opt.addConstrs((u_vars[i, j] <= alpha_vars*sp_robot.U.upper_bounds[j] for j in range(2)))

    # constraints
    opt.addConstrs((x_vars[i+1,:] == sp_robot.step(x_vars[i, :], u_vars[i, :], dt) for i in range(N-1)))
    opt.addConstrs((x_vars[i, :] >= World.lower_bounds for i in range(N)))
    opt.addConstrs((x_vars[i, :] <= World.upper_bounds for i in range(N)))

    opt.addConstr(x_vars[0, :] >= X0.lower_bounds)
    opt.addConstr(x_vars[0, :] <= X0.upper_bounds)
    opt.addConstr(x_vars[-1, :] >= Xf.lower_bounds)
    opt.addConstr(x_vars[-1, :] <= Xf.upper_bounds)
    opt.addConstr(x_vars[int(N/2), :] >= XA.lower_bounds)
    opt.addConstr(x_vars[int(N/2), :] <= XA.upper_bounds)

    # for obs in Obs:
    #     z_vars = opt.addMVar((N, 4), vtype=gp.GRB.BINARY, name="z")
    #     for i in range(N):
    #         # x_vars[i,0:2] should be outside one of the faces of the obstacle
    #         opt.addConstr(x_vars[i,0] <= obs.lower_bounds[0] + bigM*(1-z_vars[i,0]))
    #         opt.addConstr(x_vars[i,0] >= obs.upper_bounds[0] - bigM*(1-z_vars[i,1]))
    #         opt.addConstr(x_vars[i,1] <= obs.lower_bounds[1] + bigM*(1-z_vars[i,2]))
    #         opt.addConstr(x_vars[i,1] >= obs.upper_bounds[1] - bigM*(1-z_vars[i,3]))
    #         opt.addConstr(gp.quicksum(z_vars[i,:]) >= 1)

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

t_sp = np.linspace(0, N*dt, N)

# plot the trajectory
fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].plot(x_ff[:, 0], x_ff[:, 1], 'g-')
X0.plot(axs[0], color='green', alpha=0.5)
Xf.plot(axs[0], color='green', alpha=0.5)
XA.plot(axs[0], color='blue', alpha=0.5)
# [obs.plot(axs[0], color='red', alpha=0.5) for obs in Obs]
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
    u_fbl = np.linalg.pinv(uw_robot.gx(x_uw)[[0,1,6,7],0:2])@(sp_robot.fx(x) + sp_robot.gx(x)@v - uw_robot.fx(x_uw)[[0,1,6,7]])
    u_fbl = np.concatenate((u_fbl, np.zeros((4,1))))
    return u_fbl

def u_fbl_cs(x, v):
    x_uw = cs.blockcat([[x[0], x[1], 0, 0, 0, 0, x[2], x[3], 0, 0, 0, 0]])
    v_uw = cs.blockcat([[v[0], v[1], 0, 0, 0, 0]])
    fx = uw_robot.fx(x_uw.T)[[0,1,6,7]]
    gx = uw_robot.gx(x_uw.T)[[0,1,6,7],0:2]
    gx_inv = cs.DM(np.linalg.pinv(np.array(gx)))
    u_fbl = cs.mtimes(gx_inv, sp_robot.fx(x) + sp_robot.gx(x)@v - fx)
    return u_fbl

x = copy.deepcopy(x_ff[0, :])
x_hist = np.zeros((N, sp_robot.n_x))
u_hist = np.zeros((N, sp_robot.n_u))
for i in range(N):
    u = u_fbl(x, u_ff[i, :])[0:2]
    x = uw_robot.step(np.concatenate((x[0:2],np.zeros((4,)),x[2:4],np.zeros((4,)))), 
                      np.concatenate((u,np.zeros((4,1)))), dt)[[0,1,6,7]]
    x = np.array(x).squeeze()
    u = np.array(u).squeeze()
    x_hist[i, :] = x
    u_hist[i, :] = u

fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].plot(x_hist[:, 0], x_hist[:, 1], 'g-')
X0.plot(axs[0], color='green', alpha=0.5)
Xf.plot(axs[0], color='green', alpha=0.5)
XA.plot(axs[0], color='blue', alpha=0.5)
# [obs.plot(axs[0], color='red', alpha=0.5) for obs in Obs]
axs[0].set_aspect('equal', adjustable='box')
axs[1].plot(u_hist[:,0])
axs[1].plot(u_hist[:,1])
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Control input')
plt.savefig("figures/sp_trajectory_uw.png")


u_max = np.max(np.abs(u_hist), axis=0)
print(f"Alpha for uw: {np.max(u_max/uw_robot.U.upper_bounds[0:2])}")

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
        opt.addConstrs((u_vars[i, j] >= alpha*(uw_robot.U.lower_bounds[j]-u_fbl(x_vars[i,:],u_vars[i,:])[j]) for j in range(2)))
        opt.addConstrs((u_vars[i, j] <= alpha*(uw_robot.U.upper_bounds[j]-u_fbl(x_vars[i,:],u_vars[i,:])[j]) for j in range(2)))

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
elif True:
    # gradient-based optimization with casadi
    ocp = cs.Opti()
    x_vars = ocp.variable(4, N)
    u_vars = ocp.variable(2, N)
    dt_vars = ocp.variable(1)

    ocp.set_initial(x_vars, x_hist.T)
    ocp.set_initial(u_vars, u_hist.T)
    ocp.set_initial(dt_vars, 0.01)

    ocp.subject_to(dt_vars >= 0)

    for i in range(N):
        ocp.subject_to(u_fbl_cs(x_vars[:,i],u_vars[:,i])[0:2] >= alpha*(uw_robot.U.lower_bounds[0:2]))
        ocp.subject_to(u_fbl_cs(x_vars[:,i],u_vars[:,i])[0:2] <= alpha*(uw_robot.U.upper_bounds[0:2]))

    # constraints
    for i in range(N-1):
        ocp.subject_to(x_vars[:,i+1] == sp_robot.step(x_vars[:,i], u_vars[:,i], dt_vars))
    
    ocp.subject_to(x_vars[:,0] >= X0.lower_bounds)
    ocp.subject_to(x_vars[:,0] <= X0.upper_bounds)
    ocp.subject_to(x_vars[:,-1] >= Xf.lower_bounds)
    ocp.subject_to(x_vars[:,-1] <= Xf.upper_bounds)
    ocp.subject_to(x_vars[:,int(N/2)] >= XA.lower_bounds)
    ocp.subject_to(x_vars[:,int(N/2)] <= XA.upper_bounds)

    #TODO: add obstacle avoidance constraints

    # objective
    cost_eq = dt_vars
    # for i in range(N):
    #     cost_eq += (u_vars[:,i]).T@(u_vars[:,i])
    ocp.minimize(cost_eq)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'verbose':False, 'ipopt.tol': 1e-6, 'ipopt.max_iter': 1000}
    ocp.solver('ipopt',opts)

    try:
        sol = ocp.solve()
        x_uw = sol.value(x_vars).T
        u_uw = sol.value(u_vars).T
        dt_uw = sol.value(dt_vars)
        print(f"Solution found")
    except Exception as e:
        print(f"Solution not found: {e}")
        print(e)
        x_uw = np.zeros_like(x_ff)
        u_uw = np.zeros_like(u_ff)
        dt_uw = dt
else:
    # load the solution
    data = np.load("solutions/uw_solution.npz")
    x_uw = data['x_uw']
    u_uw = data['u_uw']
    dt_uw = data['dt_uw']

t_uw = np.linspace(0, dt_uw*N, N)
u_uw_fbl = np.array([u_fbl(x_uw[i,:],u_uw[i,:])[0:2].squeeze() for i in range(N)])
u_max = np.max(np.abs(u_uw_fbl), axis=0)
print(f"Replanned alpha for uw: {np.max([u_max/(uw_robot.U.upper_bounds[0:2]) for i in range(N)])}")
print(f"Replanned dt for uw: {dt_uw}")

# plot the trajectory
fig, axs = plt.subplots(2,2, figsize=(10, 10))
axs[0,0].plot(x_ff[:, 0], x_ff[:, 1], 'g-', label='sp trajectory')
axs[0,0].plot(x_uw[:, 0], x_uw[:, 1], 'c-', label='uw trajectory')
X0.plot(axs[0,0], color='green', alpha=0.5)
Xf.plot(axs[0,0], color='green', alpha=0.5)
XA.plot(axs[0,0], color='blue', alpha=0.5)
axs[0,0].legend()
axs[0,0].grid()
axs[0,0].set_aspect('equal', adjustable='box')

axs[0,1].plot(t_sp, x_ff[:, 0], 'g-', label='sp trajectory')
axs[0,1].plot(t_sp, x_ff[:, 1], 'g--')
axs[0,1].plot(t_uw, x_uw[:, 0], 'c-', label='uw trajectory')
axs[0,1].plot(t_uw, x_uw[:, 1], 'c--')
axs[0,1].set_xlabel('Time step')
axs[0,1].set_ylabel('Position')
axs[0,1].grid()
axs[0,1].legend()

axs[1,0].plot(t_sp, x_ff[:, 2], 'g-', label='sp trajectory')
axs[1,0].plot(t_sp, x_ff[:, 3], 'g--')
axs[1,0].plot(t_uw, x_uw[:, 2], 'c-', label='uw trajectory')
axs[1,0].plot(t_uw, x_uw[:, 3], 'c--')
axs[1,0].set_xlabel('Time step')
axs[1,0].set_ylabel('Velocity')
axs[1,0].grid()
axs[1,0].legend()

# axs[1,1].plot(t_uw,u_uw[:,0],'r',label='effective sp input')
# axs[1,1].plot(t_uw,u_uw[:,1],'r--')
axs[1,1].plot(t_sp,u_ff[:,0],'g',label='effective sp input')
axs[1,1].plot(t_sp,u_ff[:,1],'g--')
axs[1,1].axhline(sp_robot.U.lower_bounds[0], color='g', linestyle='-.')
axs[1,1].axhline(sp_robot.U.upper_bounds[0], color='g', linestyle='-.')
axs[1,1].plot(t_uw,u_uw_fbl[:,0],'c',label='effective uw input')
axs[1,1].plot(t_uw,u_uw_fbl[:,1],'c--')
axs[1,1].axhline(uw_robot.U.lower_bounds[0], color='c', linestyle='-.')
axs[1,1].axhline(uw_robot.U.upper_bounds[0], color='c', linestyle='-.')
axs[1,1].set_xlabel('Time step')
axs[1,1].set_ylabel('Control input')
axs[1,1].grid()
axs[1,1].legend()

plt.tight_layout()
plt.savefig("figures/sp_uw_trajectory.png")
