import numpy as np
import matplotlib.pyplot as plt
import copy

# reach-avoid gurobi optimization problem
# might want to add integer variables later so we can use gurobi to solve the problem
import gurobipy as gp
scs = {getattr(gp.GRB.status,k): k for k in dir(gp.GRB.status) if k[0].isupper()}

import casadi as cs
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from Utilities.Robots import LinearFreeFlyer2DoF, FreeFlyer, LinearBlueROV, BlueROV, LinearFreeFlyer6DoF
from Utilities.helpers import HyperRectangle, Polytope, euler_to_quat
from Utilities.stl import Pred, Spec, quant_parse_operator, OptProbItems


# hyperparameters
N = 20     # number of time steps
dt = 0.5    # time step size
t0 = 0      # initial time
tf = N*dt   # final time

bigM = 1e4

# Robot
sp_robot = LinearFreeFlyer6DoF()

# Specification
X0 = HyperRectangle(np.array([0, 0, 0]), np.array([0.1, 0.1, 0.1]))
Xf = HyperRectangle(np.array([0.90, 0.90, 0]), np.array([1, 1, 0.1]))
XA = HyperRectangle(np.array([0.2, 0.8, 0,  -np.pi/2-np.pi/8]), np.array([0.4, 1.0, 0.1,  -np.pi/2+np.pi/8]))
Obs = [HyperRectangle(np.array([0.4, -0.1]), np.array([0.6, 0.6])),
       HyperRectangle(np.array([0.75, 0.8]), np.array([0.9, 1.1]))]
World = HyperRectangle(np.array([0, 0, -10, -10]), np.array([1, 1, 10, 10]))

phi = Pred("AND", preds=[
    Pred("G", [t0,t0], preds=[Pred("MU", preds=[Polytope(X0)], dims=[0,1,2])]),
    Pred("G", [tf,tf], preds=[Pred("MU", preds=[Polytope(Xf)], dims=[0,1,2])]),
    Pred("F", [t0,tf], preds=[Pred("MU", preds=[Polytope(XA)], dims=[0,1,2, 5])]),
    Pred("G", [t0,tf], preds=[Pred("MU", preds=[Polytope(World)], dims=[0,1,6,7])]),
    # Pred("G", [t0,tf], preds=[Pred("NEG", preds=[Pred("MU", preds=[Polytope(Obs[0])], dims=[0,1])])]),
    # Pred("G", [t0,tf], preds=[Pred("NEG", preds=[Pred("MU", preds=[Polytope(Obs[1])], dims=[0,1])])])
])
spec = Spec(phi, t0, tf)


# create planner
if True:
    opt = gp.Model("prob1")
    x_vars = opt.addMVar((N, sp_robot.n_x), lb=-np.inf, ub=np.inf, name="X")
    u_vars = opt.addMVar((N, sp_robot.n_u), lb=-np.inf, ub=np.inf, name="U")
    times = np.linspace(0, tf, N)
    alpha_vars = opt.addVar(lb=0, ub=1, name="alpha")
    opt.update()

    items = OptProbItems(x_vars, u_vars, times)

    for i in range(N):
        opt.addConstrs((u_vars[i, j] >= alpha_vars*sp_robot.U_effective.lower_bounds[j] for j in range(2)))
        opt.addConstrs((u_vars[i, j] <= alpha_vars*sp_robot.U_effective.upper_bounds[j] for j in range(2)))

    # constraints
    opt.addConstrs((x_vars[i+1,:] == sp_robot.step(x_vars[i, :], u_vars[i, :], dt) for i in range(N-1)))
    sp_robot.add_state_constraints(opt, items)

    # initial orientation and velocity constraints
    opt.addConstr(x_vars[0, 3::] == np.zeros(sp_robot.n_x - 3))
    opt.addConstr(x_vars[-1, 3::] >= np.zeros(sp_robot.n_x - 3))

    X = [var for var in opt.getVars() if "X" in var.VarName]
    quant_parse_operator(opt, spec.phi, items)

    cost_var = opt.addVar(lb=-np.inf, ub=np.inf, name="cost")
    act_abs_var = opt.addMVar((N, sp_robot.n_u))
    act_int_var = opt.addVar()
    opt.addConstrs((act_abs_var[i,j] == gp.abs_(u_vars[i,j]) for i in range(N) for j in range(sp_robot.n_u)), name="act_int")
    opt.addConstr(act_int_var == gp.quicksum([act_abs_var[i,j] for i in range(N) for j in range(sp_robot.n_u)]), name="act_int_sum")
    opt.addConstr(cost_var == 1*alpha_vars - 10000*spec.phi.rho + 0.0001*act_int_var) # quad_cost
    opt.setObjective(cost_var, gp.GRB.MINIMIZE)
    opt.setParam('OutputFlag', 0)  # Suppress Gurobi output
    opt.optimize()
    if opt.status == gp.GRB.OPTIMAL:
        print(f"Optimal solution found")
        print(f"\nAlpha from solving Prob 1: {alpha_vars.X}")
        denom = sp_robot.U_effective.scalar_multiply(alpha_vars.X)
        denom = denom.sum(sp_robot.KD)
        print(f"Beta from solving Prob 1: {denom.divide(sp_robot.U)}")
        print(f"Spatial robustness: rho = {spec.phi.rho.X}")
        print(f"Control integral summed: {act_int_var.X}")
        # print(f"Spatial robustness: {spec.phi.rho.X}")
        # for pred in spec.phi.preds:
        #     print(f"rhos: {pred.preds[-1].rhos.X }")
        # print(f"rhos: {[rf.X for rf in spec.phi.preds[-1].preds[0].preds[0].rho_faces]}")
        
        print(f"This is how much control is necessary to satisfy the specification")
    else:
        print(f"No optimal solution found")
        print(f"Status: {scs[opt.status]}")

    x_ff = x_vars.X
    u_ff = u_vars.X
    alpha = alpha_vars.X
    # save x_ff and u_ff to a csv file
    np.savez('Planning/solutions/sp_solution.npz', x_ff=x_ff, u_ff=u_ff, dt=dt, alpha=alpha, times=times)

else:
    # or load instead
    data = np.load('Planning/solutions/sp_solution.npz')
    x_ff = data['x_ff']
    u_ff = data['u_ff']
    alpha = data['alpha']
    dt = data['dt']
    # t_sp = data['times']

# plot the trajectory
fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].plot(x_ff[:, 0], x_ff[:, 1], 'g-')
axs[0].plot(x_ff[:, 0], x_ff[:, 1], 'go')
X0.plot(axs[0], color='green', alpha=0.5)
Xf.plot(axs[0], color='green', alpha=0.5)
XA.plot(axs[0], color='blue', alpha=0.5)
[obs.plot(axs[0], color='red', alpha=0.5) for obs in Obs]
axs[0].set_aspect('equal', adjustable='box')

axs[1].plot(u_ff[:,0])
axs[1].plot(u_ff[:,1])
axs[1].axhline(sp_robot.U.lower_bounds[0], color='g', linestyle='-.', label="U_lb")
axs[1].axhline(sp_robot.U.upper_bounds[0], color='g', linestyle='-.')
axs[1].axhline(alpha*sp_robot.U_effective.lower_bounds[0], color='b', linestyle='--', label="alpha*U_effective_lb")
axs[1].axhline(alpha*sp_robot.U_effective.upper_bounds[0], color='b', linestyle='--')
axs[1].axhline(sp_robot.U_effective.lower_bounds[0], color='r', linestyle=':', label="U_effective_lb")
axs[1].axhline(sp_robot.U_effective.upper_bounds[0], color='r', linestyle=':')
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Control input')
plt.savefig("figures/sp_trajectory.png")



# Now we have pitch-roll-yaw angles which we want to convert to unit quaternion
x_ff_quat = np.zeros((x_ff.shape[0], x_ff.shape[1] + 1))
for i in range(x_ff.shape[0]):
    x_ff_quat[i, 0:3] = x_ff[i, 0:3]
    x_ff_quat[i, 3:7] = euler_to_quat(x_ff[i, 3], x_ff[i, 4], x_ff[i, 5])
    x_ff_quat[i, 7::] = x_ff[i, 6::]

print(x_ff_quat[:, 3:7])
print(np.linalg.norm(x_ff_quat[:, 3:7],axis=1))  # should be close to 1

# feedback linearization controller for the underwater robot to behave like a free flyer
uw_robot = BlueROV()

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
x_sp = np.zeros((N, sp_robot.n_x))
u_sp = np.zeros((N, sp_robot.n_u))
for i in range(N):
    u = u_fbl(x, u_ff[i, :])[0:2]
    x = uw_robot.step(np.concatenate((x[0:2],np.zeros((4,)),x[2:4],np.zeros((4,)))), 
                      np.concatenate((u,np.zeros((4,1)))), dt)[[0,1,6,7]]
    x = np.array(x).squeeze()
    u = np.array(u).squeeze()
    x_sp[i, :] = x
    u_sp[i, :] = u

fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].plot(x_sp[:, 0], x_sp[:, 1], 'g-')
X0.plot(axs[0], color='green', alpha=0.5)
Xf.plot(axs[0], color='green', alpha=0.5)
XA.plot(axs[0], color='blue', alpha=0.5)
# [obs.plot(axs[0], color='red', alpha=0.5) for obs in Obs]
axs[0].set_aspect('equal', adjustable='box')
axs[1].plot(u_sp[:,0])
axs[1].plot(u_sp[:,1])
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Control input')
plt.savefig("figures/sp_trajectory_uw.png")


u_max = np.max(np.abs(u_sp), axis=0)
print(f"\nAlpha if u_sp applied to BlueROV directly: {np.max(u_max/uw_robot.U.upper_bounds[0:2])}")
print(f"This is lower because BlueROV is faster than free flyer")

u_uw = np.array([u_fbl(x_sp[i,:],u_sp[i,:])[0:2].squeeze() for i in range(N)])
u_max = np.max(np.abs(u_uw), axis=0)
print(f"\nAlpha if u_fbl(x_sp,u_sp) applied to BlueROV directly: {np.max(u_max/uw_robot.U.upper_bounds[0:2])}")
print(f"U should become {np.max(u_max)/alpha} for same alpha (was {uw_robot.U.upper_bounds[0]}) which is {(np.max(u_max)/alpha)/uw_robot.U.upper_bounds[0]*100}%")

# So now we have to find \phi (for now deltaT) such that
# alpha for uw is equal to alpha for the free flyer
# we have bilinear constraints so that is quite tricky but
# we have good initial guesses regarding obstacle avoidance
if True:
    # gradient-based optimization with casadi
    ocp = cs.Opti()
    x_vars = ocp.variable(4, N)
    u_vars = ocp.variable(2, N)
    dt_vars = ocp.variable(1)

    ocp.set_initial(x_vars, x_sp.T)
    ocp.set_initial(u_vars, u_sp.T)
    ocp.set_initial(dt_vars, 0.01)

    ocp.subject_to(dt_vars >= 0)

    for i in range(N):
        ocp.subject_to(u_fbl_cs(x_vars[:,i],u_vars[:,i])[0:2] >= alpha*(uw_robot.U.lower_bounds[0:2]))
        ocp.subject_to(u_fbl_cs(x_vars[:,i],u_vars[:,i])[0:2] <= alpha*(uw_robot.U.upper_bounds[0:2]))

    # constraints
    for i in range(N-1):
        ocp.subject_to(x_vars[:,i+1] == sp_robot.step(x_vars[:,i], u_vars[:,i], dt_vars))
    
    for i in range(N):
        ocp.subject_to(x_vars[0:2,i] == x_sp[i,0:2])
    #TODO: seems to be a singularity around i=50
    # points = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # for i in points:
    #     ocp.subject_to(x_vars[0:2,i] == x_sp[i,0:2])

    # ocp.subject_to(x_vars[:,0] >= X0.lower_bounds)
    # ocp.subject_to(x_vars[:,0] <= X0.upper_bounds)
    # ocp.subject_to(x_vars[:,-1] >= Xf.lower_bounds)
    # ocp.subject_to(x_vars[:,-1] <= Xf.upper_bounds)
    # ocp.subject_to(x_vars[:,int(N/2)] >= XA.lower_bounds)
    # ocp.subject_to(x_vars[:,int(N/2)] <= XA.upper_bounds)

    # # spatial robustness of the specification (stay-in at N/2)
    # delta = ocp.variable(1)
    # for face_idx in range(len(XA.b)):
    #     c = -cs.mtimes(np.array([XA.A[face_idx, :]]),x_vars[0:2,int(N/2)]) + XA.b[face_idx]
    #     ocp.subject_to(c >= delta)

    # objective
    cost_var = 100*dt_vars #- 10000*delta
    # for i in range(N):
    #     cost_eq += (u_vars[:,i]).T@(u_vars[:,i])
    ocp.minimize(cost_var)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'verbose':False, 'ipopt.tol': 1e-6, 'ipopt.max_iter': 1000}
    ocp.solver('ipopt',opts)

    try:
        sol = ocp.solve()
        x_uw = sol.value(x_vars).T
        u_uw = sol.value(u_vars).T
        dt_uw = sol.value(dt_vars)
        # print(f"Solution found")
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
print(f"\nAlpha for replanning uw to have same alpha: {np.max([u_max/(uw_robot.U.upper_bounds[0:2]) for i in range(N)])}")
print(f"This should be the same as the solution to Prob 1")
print(f"Replanned dt_uw: {dt_uw} (was {dt}) which is {(dt_uw)/dt*100}%")


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
