import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
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
from Utilities.Robots import FreeFlyer, LinearFreeFlyer6DoF
from Utilities.sets import HyperRectangle, Polytope
from Utilities.rotations import euler_to_quat_np, euler_to_quat_cs
from Utilities.rotations import x_ff_to_x_uw, x_uw_to_x_ff
from Utilities.stl import Pred, Spec, quant_parse_operator, OptProbItems
from Utilities.plotting import plot_planning_results
from Utilities.smarc_modelling.src.smarc_modelling.vehicles.BlueROV import BlueROV 

# hyperparameters
N = 30     # number of time steps
dt = 1.5    # time step size
t0 = 0      # initial time
tf = (N-1)*dt   # final time

bigM = 1e4

# Robot
sp_robot = LinearFreeFlyer6DoF()

euler_order = 'xyz'

# STL Specification
X0 = HyperRectangle(np.array([0.5, 0, 2.0]), np.array([0.6, 0.1, 2.1]))
Xf = HyperRectangle(np.array([2.5, 1.0, 2.0]), np.array([2.6, 1.1, 2.1]))
XA = HyperRectangle(np.array([2.5, -1.0, 2.0,  -np.pi/2-np.pi/8]), np.array([2.6, -0.9, 2.1,  -np.pi/2+np.pi/8]))
# later converted to polytopes for predicates of the form Ax \leq b: Polytope = (A,b)
# XA = HyperRectangle(np.array([2.5, -1.0, 2.0]), np.array([2.6, -0.9, 2.1]))
Obs = []
World = HyperRectangle(np.array([0, -1.5, -10, -10]), np.array([4, 1.5, 10, 10]))
phi = Pred("AND", preds=[
    Pred("G", [t0,t0], preds=[Pred("MU", preds=[Polytope(X0)], dims=[0,1,2])]),
    Pred("G", [tf,tf], preds=[Pred("MU", preds=[Polytope(Xf)], dims=[0,1,2])]),
    Pred("F", [t0,tf], preds=[Pred("MU", preds=[Polytope(XA)], dims=[0,1,2,5])]),
    Pred("G", [t0,tf], preds=[Pred("MU", preds=[Polytope(World)], dims=[0,1,6,7])]),
])
spec = Spec(phi, t0, tf)

# Initial Motion planner on Linear Model
if True:
    opt = gp.Model("prob1")
    x_vars = opt.addMVar((N, sp_robot.n_x), lb=-np.inf, ub=np.inf, name="X")
    u_vars = opt.addMVar((N-1, sp_robot.n_u), lb=-np.inf, ub=np.inf, name="U")
    t_sp = np.linspace(0,(N-1)*dt, N)
    alpha_vars = opt.addVar(lb=0, ub=1, name="alpha")
    opt.update()

    items = OptProbItems(x_vars, u_vars, t_sp)

    for i in range(N-1):
        opt.addConstrs((u_vars[i, j] >= alpha_vars*sp_robot.U_effective.lower_bounds[j] for j in range(sp_robot.n_u)))
        opt.addConstrs((u_vars[i, j] <= alpha_vars*sp_robot.U_effective.upper_bounds[j] for j in range(sp_robot.n_u)))

    # constraints
    for i in range(N-1):
        opt.addConstr(x_vars[i+1,:] == sp_robot.step(x_vars[i, :], u_vars[i, :], dt))
    sp_robot.add_state_constraints(opt, items)

    # initial orientation and velocity constraints
    opt.addConstr(x_vars[0, 3::] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))#np.zeros(sp_robot.n_x - 3))
    opt.addConstr(x_vars[-1, 3::] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))#np.zeros(sp_robot.n_x - 3))

    X = [var for var in opt.getVars() if "X" in var.VarName]
    quant_parse_operator(opt, spec.phi, items)

    cost_var = opt.addVar(lb=-np.inf, ub=np.inf, name="cost")

    # # Quadratic actuation cost
    # act_quadu_vars = opt.addMVar((N-1,))
    # act_quadu_var = opt.addVar()
    # opt.addConstrs((act_quadu_vars[i] == u_vars[i,:]@u_vars[i,:] for i in range(N-1)), name="act_int")
    # opt.addConstr(act_quadu_var == gp.quicksum([act_quadu_vars[i] for i in range(N-1)]), name="act_int_sum")

    # L1 actuation cost
    act_absu_vars = opt.addMVar((N-1,sp_robot.n_u))
    act_absu_var = opt.addVar(lb=0, ub=np.inf)
    opt.addConstrs((act_absu_vars[i, j] == gp.abs_(u_vars[i, j]) for i in range(N-1) for j in range(sp_robot.n_u)), name="act_abs")
    opt.addConstr(act_absu_var == gp.quicksum([act_absu_vars[i, j] for i in range(N-1) for j in range(sp_robot.n_u)]), name="act_abs_sum")

    # Final cost
    opt.addConstr(cost_var == 1*alpha_vars - 1000*spec.phi.rho + 0.01*act_absu_var) # quad_cost
    opt.setObjective(cost_var, gp.GRB.MINIMIZE)
    opt.setParam('OutputFlag', 0)  # Suppress Gurobi output
    opt.optimize()
    if opt.status == gp.GRB.OPTIMAL:
        print(f"Optimal solution found")
        print(f"\nAlpha from solving Prob 1: {alpha_vars.X}")
        print(f"This is how much control is necessary to satisfy the specification")
        denom = sp_robot.U_effective.scalar_multiply(alpha_vars.X)
        denom = denom.sum(sp_robot.KD)
        print(f"Beta from solving Prob 1: {denom.divide(sp_robot.U)}")
        print(f"Spatial robustness: rho = {spec.phi.rho.X}")
        # print(f"Control integral summed: {act_quadu_var.X}")
    else:
        print(f"No optimal solution found")
        print(f"Status: {scs[opt.status]}")

    x_ff = x_vars.X
    u_ff = u_vars.X
    alpha = alpha_vars.X
    # save x_ff and u_ff to a csv file
    np.savez('stl_mapping/Planning/solutions/atmos_solution_lin.npz', x=x_ff, u=u_ff, dt=dt, alpha=alpha, times=t_sp)
else:
    # or load instead
    data = np.load('stl_mapping/Planning/solutions/atmos_solution_lin.npz')
    x_ff = data['x_ff']
    u_ff = data['u_ff']
    alpha = data['alpha']
    dt = data['dt']
    t_sp = data['times']

#! Convert [p:ENU, e:FLU->ENU, v:ENU, w:FLU] to [p:ENU, q:FLU->ENU, v:ENU, w:FLU]
x_ff_converted = np.zeros((x_ff.shape[0], x_ff.shape[1] + 1))
for i in range(x_ff.shape[0]):
    x_ff_converted[i, :3] = x_ff[i, :3]
    x_ff_converted[i,3:7] = euler_to_quat_np(x_ff[i, 3:6], order=euler_order)
    x_ff_converted[i,7:10] = x_ff[i, 6:9]
    x_ff_converted[i,10:13] = x_ff[i, 9:12] #enu_to_flu(x_ff[i, 9:12], x_ff_converted[i, 3:7])
x_ff = x_ff_converted
#! Convert [F: ENU, tau: FLU] to [F: FLU, tau:FLU]
u_ff_converted = np.zeros_like(u_ff)
for i in range(u_ff.shape[0]):
    q = R.from_quat(x_ff[i, 3:7], scalar_first=True)
    u_ff_converted[i, :] = np.hstack((q.inv().apply(u_ff[i, 0:3]), u_ff[i, 3:6]))  # convert u from ENU to body FLU body frame
u_ff = u_ff_converted

# plot the trajectory
plot_planning_results(sp_robot, t_sp, x_ff, u_ff, X0, Xf, [XA], Obs, alpha=alpha,
                      path="stl_mapping/Planning/figures/atmos_trajectory.png")

np.savez('stl_mapping/Planning/solutions/atmos_solution.npz', x=x_ff, u=u_ff, dt=dt, alpha=alpha, times=t_sp)

#TODO: 1. this is a valid conversion (Lin to non-lin) if the system (with zero roll) is differentially flat?
#TODO:    this means that this linear decoupling is valid, and x_ff and u_ff are valid for the nonlinear space robot
#TODO:    CHECK THIS!
# sp_robot_nl = FreeFlyer()
# x_ff_nl = np.zeros((x_ff.shape[0], x_ff.shape[1]))
# u_ff_nl = np.zeros((u_ff.shape[0], u_ff.shape[1]))
# x_ff_i = copy.deepcopy(x_ff[0, :])
# x_ff_nl[0, :] = x_ff_i
# for i in range(x_ff.shape[0]-1):
#     x_ff_i = sp_robot_nl.step(x_ff_i, u_ff[i, :], dt)
#     x_ff_i[3:7] /= np.linalg.norm(x_ff_i[3:7])  # normalize quaternion
#     x_ff_nl[i+1, :] = x_ff_i
# plot_planning_results(sp_robot_nl, t_sp, x_ff_nl, u_ff, X0, Xf, [XA], Obs, alpha=alpha,
#                       path="stl_mapping/Planning/figures/atmos_trajectory_as_ff.png")


# feedback linearization controller for the underwater robot to behave like a free flyer
uw_robot = BlueROV()
sp_robot_nl = FreeFlyer()

#TODO: 1. fix the BlueRov model such that it behaves equal to David's model (yaw and pitch seem wrong)
#TODO: 2. fix frame conversions for feedback linearization (below)
def u_fbl(x_sp, u_sp):
    '''
    Feedback linearization control for the underwater robot
    Args:
        x: state of FF [p: ENU, q:FLU->ENU, v:ENU , w:FLU] in ENU frame
        v: control input of FF in ENU frame
    Returns:
        u_fbl: control input for the space robot in ENU body frame
    '''
    fx_sp = sp_robot_nl.calculate_fx(x_sp)
    gx_sp = sp_robot_nl.calculate_gx(x_sp)
    dx_sp = fx_sp + gx_sp@u_sp

    #! Now [dp: ENU, dq:FLfU->ENU, dv: ENU, dw:FLU] to [dp: NED, dq:FRD->NED, dv:FRD, dw:FRD] 
    dp, dq, dv, dw = dx_sp[:3], dx_sp[3:7], dx_sp[7:10], dx_sp[10:13]
    dp_ned = np.array([dp[1], dp[0], -dp[2]])
    dq_ned = 1/np.sqrt(2) * np.array([dq[0] + dq[3], dq[1] + dq[2], dq[1] - dq[2], dq[0] - dq[3]])
    dv_flu = R.from_quat(x_sp[3:7], scalar_first=True).inv().apply(dv)
    dv_frd = np.array([dv_flu[0], -dv_flu[1], -dv_flu[2]])
    dw_frd = np.array([dw[0], -dw[1], -dw[2]])
    dx_sp_ned = np.concatenate((dp_ned, dq_ned, dv_frd, dw_frd))

    x_uw = x_ff_to_x_uw(x_sp)

    fx_uw = uw_robot.calculate_fx(x_uw)
    gx_uw = uw_robot.calculate_gx(x_uw)
    u_fbl = np.linalg.pinv(gx_uw)@(dx_sp_ned - fx_uw)
    # print(f"u: {u}, u_fbl: {u_fbl}")
    return u_fbl


x_ff_i = copy.deepcopy(x_ff[0, :])
x_uw_i = x_ff_to_x_uw(x_ff_i)

x_uw_fbl_sp = np.zeros((N, sp_robot_nl.n_x))
x_uw_fbl_sp[0, :] = x_uw_i
u_uw_fbl_sp = np.zeros((N, sp_robot_nl.n_u))
for i in range(N-1):
    x_ff_i = x_uw_to_x_ff(x_uw_i)

    u_uw = u_fbl(x_ff_i, u_ff[i, :])
    x_uw_i = uw_robot.step(x_uw_i, u_uw, dt)
    x_uw_i[3:7] /= np.linalg.norm(x_uw_i[3:7])
    # save state and control
    x_uw_fbl_sp[i+1, :] = x_uw_i
    u_uw_fbl_sp[i, :] = u_uw


# save the trajectory
np.savez('stl_mapping/Planning/solutions/bluerov_solution_fbl.npz', x=x_uw_fbl_sp, u=u_uw_fbl_sp, dt=dt, alpha=alpha, times=t_sp)

# plot the trajectory
plot_planning_results(uw_robot, t_sp, x_uw_fbl_sp, u_uw_fbl_sp, X0, Xf, [XA], Obs, alpha=alpha,
                      path="stl_mapping/Planning/figures/atmos_trajectory_as_bluerov.png")

# Analysis of alpha
u_max = np.max(np.abs(u_uw_fbl_sp), axis=0)
print(f"\nAlpha if u_fbl(x_sp,u_sp) applied to BlueROV directly: {np.max(u_max/uw_robot.U.upper_bounds)} (was {alpha} for ff)")
print(f"This is lower because BlueROV is faster than free flyer")
print(f"U should become {np.max(u_max)/alpha} for same alpha (was {uw_robot.U.upper_bounds[0]}) which is {(np.max(u_max)/alpha)/uw_robot.U.upper_bounds[0]*100}%")










# uw_robot = BlueROV(iX=cs.SX)
# # call compute_K a bunch of times with different states to check what the values are
# for i in range(1):
#     print(f"D: {uw_robot.D.lower_bounds}")
#     print(f"U: {uw_robot.U.lower_bounds}")
#     # print(f"U: {uw_robot.U}")
#     print(f"gx: {uw_robot.calculate_gx(np.random.rand(13))}")

#     print(f"K: {uw_robot.calculate_K(np.random.rand(13))}")

#TODO: now we either solve the time-scaling of trajectory and control input
#TODO: or we solve the optimization problem to make alpha equal, we need to figure out what is warranted
#TODO: - time-scaling?
#TODO: - control bound scaling? (only valid for faster systems, but still, is it valid?)
#TODO: - trajectory optimization? (this should be valid, but perhaps dirtier than warranted)
uw_robot = BlueROV(iX=cs.MX)
sp_robot_nl = FreeFlyer()
if True:
    # gradient-based optimization with casadi
    ocp = cs.Opti()
    x_vars = ocp.variable(13, N)
    u_vars = ocp.variable(6, N)
    dt_vars = ocp.variable(1)
    delta = ocp.variable(1)

    ocp.set_initial(x_vars, x_uw_fbl_sp.T)
    ocp.set_initial(u_vars, u_uw_fbl_sp.T)
    ocp.set_initial(dt_vars, 0.5)

    ocp.subject_to(0.1 <= dt_vars)
    ocp.subject_to(dt_vars <= dt)
    ocp.subject_to(delta >= 0)

    for i in range(N):
        # ocp.subject_to(u_vars[:,i] >= alpha * uw_robot.U.lower_bounds)
        # ocp.subject_to(u_vars[:,i] <= alpha * uw_robot.U.upper_bounds)
        ocp.subject_to(u_vars[:,i] >= alpha * uw_robot.calculate_U_effective(x_vars[:,i]).lower_bounds)
        ocp.subject_to(u_vars[:,i] <= alpha * uw_robot.calculate_U_effective(x_vars[:,i]).upper_bounds)
        # ocp.subject_to(u_vars[:,i] >= u_fbl_cs(x_vars[:,i], alpha*uw_robot.calculate_U_effective(x_vars[:,i]).lower_bounds).squeeze())
        # ocp.subject_to(u_vars[:,i] <= u_fbl_cs(x_vars[:,i], alpha*uw_robot.calculate_U_effective(x_vars[:,i]).upper_bounds).squeeze())

    # constraints
    for i in range(N-1):
        ocp.subject_to(x_vars[:,i+1] == uw_robot.step(x_vars[:,i], u_vars[:,i], dt_vars))
    
    # enforce position equality with the free flyer
    n_equal = 7 # first n_equal states: [p,q]
    for i in range(N):
        # ocp.subject_to(x_vars[0:n_equal,i] == x_uw_fbl_sp[i,0:n_equal])
        ocp.subject_to(x_vars[0:n_equal,i] <= x_uw_fbl_sp[i,0:n_equal] + delta* np.ones(n_equal))
        ocp.subject_to(x_vars[0:n_equal,i] >= x_uw_fbl_sp[i,0:n_equal] - delta* np.ones(n_equal))

    # objective
    cost_var = dt_vars + 1e5 * delta
    ocp.minimize(cost_var)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'verbose':False, 'ipopt.tol': 1e-6, 'ipopt.max_iter': 1000}
    ocp.solver('ipopt',opts)

    try:
        print(f"Solving optimization problem...")
        sol = ocp.solve()
        x_uw = sol.value(x_vars).T
        u_uw = sol.value(u_vars).T
        dt_uw = sol.value(dt_vars)
        t_uw = np.linspace(0, (N-1)*dt_uw, N)
        print(f"delta: {sol.value(delta)}")
        print(f"dt_uw: {dt_uw}")
        # print(f"Solution found")
    except Exception as e:
        print(f"Solution not found: {e}")
        print(e)
        x_uw = np.zeros_like(x_ff)
        u_uw = np.zeros_like(u_ff)
        t_uw = np.linspace(0, (N-1)*dt, N)
        dt_uw = dt
else:
    # load the solution
    data = np.load("solutions/bluerov_solution.npz")
    x_uw = data['x_uw']
    u_uw = data['u_uw']
    dt_uw = data['dt_uw']

# t_uw = np.linspace(0, dt_uw*N, N)
# u_uw_fbl = np.array([u_fbl(x_uw[i,:],u_uw[i,:])[0:2].squeeze() for i in range(N)])
# u_max = np.max(np.abs(u_uw_fbl), axis=0)
# print(f"\nAlpha for replanning uw to have same alpha: {np.max([u_max/(uw_robot.U.upper_bounds[0:2]) for i in range(N)])}")
# print(f"This should be the same as the solution to Prob 1")
# print(f"Replanned dt_uw: {dt_uw} (was {dt}) which is {(dt_uw)/dt*100}%")

# save the trajectory
np.savez('stl_mapping/Planning/solutions/bluerov_solution.npz', x=x_uw, u=u_uw, dt=dt_uw, alpha=alpha, times=t_uw)

# plot the trajectory
plot_planning_results(uw_robot, t_uw, x_uw, u_uw, X0, Xf, [XA], Obs, alpha=alpha,
                      path="stl_mapping/Planning/figures/bluerov_trajectory.png")
