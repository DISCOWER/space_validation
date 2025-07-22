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
from Utilities.rotations import quat_to_euler_np, quat_to_euler_cs, euler_to_quat_np, euler_to_quat_cs
from Utilities.rotations import quat_x_to_euler_x_cs, euler_x_to_quat_x_cs, quat_x_to_euler_x_np, euler_x_to_quat_x_np
from Utilities.rotations import enu_to_ned, ned_to_enu, u_enu_to_ned
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

# # Specification
# X0 = HyperRectangle(np.array([0, 0, 0]), np.array([0.1, 0.1, 0.1]))
# Xf = HyperRectangle(np.array([0.90, 0.90, 0]), np.array([1, 1, 0.1]))
# XA = HyperRectangle(np.array([0.2, 0.8, 0,  -np.pi/2-np.pi/8]), np.array([0.4, 1.0, 0.1,  -np.pi/2+np.pi/8]))
# # XA = HyperRectangle(np.array([0.2, 0.8, 0]), np.array([0.4, 1.0, 0.1]))
# Obs = [HyperRectangle(np.array([0.4, -0.1]), np.array([0.6, 0.6])),
#        HyperRectangle(np.array([0.75, 0.8]), np.array([0.9, 1.1]))]
# World = HyperRectangle(np.array([0, 0, -10, -10]), np.array([1, 1, 10, 10]))
# phi = Pred("AND", preds=[
#     Pred("G", [t0,t0], preds=[Pred("MU", preds=[Polytope(X0)], dims=[0,1,2])]),
#     Pred("G", [tf,tf], preds=[Pred("MU", preds=[Polytope(Xf)], dims=[0,1,2])]),
#     Pred("F", [t0,tf], preds=[Pred("MU", preds=[Polytope(XA)], dims=[0,1,2, 5])]),
#     # Pred("F", [t0,tf], preds=[Pred("MU", preds=[Polytope(XA)], dims=[0,1,2])]),
#     Pred("G", [t0,tf], preds=[Pred("MU", preds=[Polytope(World)], dims=[0,1,6,7])]),
#     # Pred("G", [t0,tf], preds=[Pred("NEG", preds=[Pred("MU", preds=[Polytope(Obs[0])], dims=[0,1])])]),
#     # Pred("G", [t0,tf], preds=[Pred("NEG", preds=[Pred("MU", preds=[Polytope(Obs[1])], dims=[0,1])])])
# ])
# spec = Spec(phi, t0, tf)

euler_order = 'zyx'

X0 = HyperRectangle(np.array([0.5, 0, 0]), np.array([0.6, 0.1, 0.1]))
Xf = HyperRectangle(np.array([2.5, 1.0, 0]), np.array([2.6, 1.1, 0.1]))
XA = HyperRectangle(np.array([2.5, -1.0, 0,  -np.pi/2-np.pi/8]), np.array([2.6, -0.9, 0.1,  -np.pi/2+np.pi/8]))
# XA = HyperRectangle(np.array([2.5, -1.0, 0]), np.array([2.6, -0.9, 0.1]))
Obs = []
World = HyperRectangle(np.array([0, -1.5, -10, -10]), np.array([4, 1.5, 10, 10]))
phi = Pred("AND", preds=[
    Pred("G", [t0,t0], preds=[Pred("MU", preds=[Polytope(X0)], dims=[0,1,2])]),
    Pred("G", [tf,tf], preds=[Pred("MU", preds=[Polytope(Xf)], dims=[0,1,2])]),
    Pred("F", [t0,tf], preds=[Pred("MU", preds=[Polytope(XA)], dims=[0,1,2, 3])]), # 3: pitch, 4: roll, 5: yaw
    Pred("G", [t0,tf], preds=[Pred("MU", preds=[Polytope(World)], dims=[0,1,6,7])]),
])
spec = Spec(phi, t0, tf)


# create planner
if False:
    opt = gp.Model("prob1")
    x_vars = opt.addMVar((N, sp_robot.n_x), lb=-np.inf, ub=np.inf, name="X")
    u_vars = opt.addMVar((N, sp_robot.n_u), lb=-np.inf, ub=np.inf, name="U")
    t_sp = np.linspace(0,(N-1)*dt, N)
    alpha_vars = opt.addVar(lb=0, ub=1, name="alpha")
    opt.update()

    items = OptProbItems(x_vars, u_vars, t_sp)

    for i in range(N):
        opt.addConstrs((u_vars[i, j] >= alpha_vars*sp_robot.U_effective.lower_bounds[j] for j in range(sp_robot.n_u)))
        opt.addConstrs((u_vars[i, j] <= alpha_vars*sp_robot.U_effective.upper_bounds[j] for j in range(sp_robot.n_u)))

    # constraints
    opt.addConstrs((x_vars[i+1,:] == sp_robot.step(x_vars[i, :], u_vars[i, :], dt) for i in range(N-1)))
    sp_robot.add_state_constraints(opt, items)

    # initial orientation and velocity constraints
    opt.addConstr(x_vars[0, 3::] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))#np.zeros(sp_robot.n_x - 3))
    opt.addConstr(x_vars[-1, 3::] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))#np.zeros(sp_robot.n_x - 3))

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
        print(f"This is how much control is necessary to satisfy the specification")
        denom = sp_robot.U_effective.scalar_multiply(alpha_vars.X)
        denom = denom.sum(sp_robot.KD)
        print(f"Beta from solving Prob 1: {denom.divide(sp_robot.U)}")
        print(f"Spatial robustness: rho = {spec.phi.rho.X}")
        print(f"Control integral summed: {act_int_var.X}")
    else:
        print(f"No optimal solution found")
        print(f"Status: {scs[opt.status]}")

    x_ff = x_vars.X
    u_ff = u_vars.X
    alpha = alpha_vars.X
    # save x_ff and u_ff to a csv file
    np.savez('stl_mapping/Planning/solutions/sp_solution_euler.npz', x_ff=x_ff, u_ff=u_ff, dt=dt, alpha=alpha, times=t_sp)
else:
    # or load instead
    data = np.load('stl_mapping/Planning/solutions/sp_solution_euler.npz')
    x_ff = data['x_ff']
    u_ff = data['u_ff']
    alpha = data['alpha']
    dt = data['dt']
    t_sp = data['times']

# plot the trajectory
plot_planning_results(sp_robot, t_sp, x_ff, u_ff, X0, Xf, [XA], Obs, alpha=alpha,
                      path="stl_mapping/Planning/figures/sp_trajectory.png")


# Now we have pitch-roll-yaw angles which we want to convert to unit quaternion
x_ff_quat = np.zeros((x_ff.shape[0], x_ff.shape[1] + 1))
for i in range(x_ff.shape[0]):
    x_ff_quat[i,:] = euler_x_to_quat_x_np(x_ff[i, :],order=euler_order)
print(f"euler: {x_ff[:, 3:6]}")
print(f"quat:  {x_ff_quat[:, 3:7]}")
print(f"euler: {np.array([quat_to_euler_np(x[3:7], order=euler_order) for x in x_ff_quat])}")

np.savez('stl_mapping/Planning/solutions/sp_solution_quat.npz', x_ff=x_ff_quat, u_ff=u_ff, dt=dt, alpha=alpha, times=t_sp)

# feedback linearization controller for the underwater robot to behave like a free flyer
uw_robot = BlueROV() # (quaternion=True)
sp_robot_nl = FreeFlyer()

def u_fbl(x, v):
    # convert p and q from ENU to NED
    x_ned = enu_to_ned(x[:3], x[3:7])
    # convert dq and dp from FRD body frame to NED inertial frame
    dp_ned = R.from_quat(x[3:7],scalar_first=True).apply(x[7:10])
    # convert control input v from ENU to NED
    v_ned = u_enu_to_ned(v)

    fx_uw = uw_robot.fx(x)
    gx_uw = uw_robot.gx(x)
    u_fbl = np.linalg.pinv(gx_uw)@(sp_robot_nl.fx(x) + sp_robot_nl.gx(x)@v - fx_uw)
    return u_fbl
def u_fbl_cs(x, v):
    fx_uw = uw_robot.fx(x)
    gx_uw = uw_robot.gx(x)
    u_fbl = cs.mtimes(cs.pinv(gx_uw), sp_robot_nl.fx(x) + cs.mtimes(sp_robot_nl.gx(x),v) - fx_uw)
    return u_fbl

x = copy.deepcopy(x_ff_quat[0, :])
x_uw_fbl_sp = np.zeros((N, sp_robot_nl.n_x))
x_uw_fbl_sp[0, :] = x
u_uw_fbl_sp = np.zeros((N, sp_robot_nl.n_u))
for i in range(N-1):
    u = u_fbl(x, u_ff[i, :])
    x = uw_robot.step(x, u, dt)
    x = np.array(x).squeeze()
    u = np.array(u).squeeze()
    x_uw_fbl_sp[i+1, :] = x
    u_uw_fbl_sp[i, :] = u

# convert to euler angles for intuitive plotting
x_uw_fbl_sp_euler = np.zeros((x_uw_fbl_sp.shape[0], x_uw_fbl_sp.shape[1] - 1))
for i in range(x_uw_fbl_sp.shape[0]):
    x_uw_fbl_sp_euler[i, :] = quat_x_to_euler_x_np(x_uw_fbl_sp[i, :], order=euler_order)

# plot the trajectory
plot_planning_results(uw_robot, t_sp, x_uw_fbl_sp_euler, u_uw_fbl_sp, X0, Xf, [XA], Obs, alpha=alpha,
                      path="stl_mapping/Planning/figures/sp_trajectory_uw.png")

# Analysis of alpha
u_max = np.max(np.abs(u_uw_fbl_sp), axis=0)
print(f"\nAlpha if u_sp applied to BlueROV directly: {np.max(u_max/uw_robot.U.upper_bounds)}")
print(f"This is lower because BlueROV is faster than free flyer")

u_uw = np.array([u_fbl(x_uw_fbl_sp[i,:],u_uw_fbl_sp[i,:]) for i in range(N)]).squeeze()
u_max = np.max(np.abs(u_uw), axis=0)
print(f"\nAlpha if u_fbl(x_sp,u_sp) applied to BlueROV directly: {np.max(u_max/uw_robot.U.upper_bounds)}")
print(f"U should become {np.max(u_max)/alpha} for same alpha (was {uw_robot.U.upper_bounds[0]}) which is {(np.max(u_max)/alpha)/uw_robot.U.upper_bounds[0]*100}%")


# # test if different backends are equal
# uw_robot_cs = BlueROV(backend='cs')
# uw_robot_np = BlueROV(backend='np')
# for i in range(100):
#     x = np.random.rand(13)
#     u = np.random.rand(6)
#     x_cs = cs.DM(x)
#     u_cs = cs.DM(u)
#     x_uw_cs = uw_robot_cs.step(x_cs, u_cs, dt)
#     x_uw_np = uw_robot_np.step(x, u, dt)
#     print(f"Step {i}: {x_uw_cs} == {x_uw_np}")


# So now we have to find \phi (for now deltaT) such that
# alpha for uw is equal to alpha for the free flyer
# we have bilinear constraints so that is quite tricky but
# we have good initial guesses regarding obstacle avoidance
uw_robot = BlueROV(backend='cs')
sp_robot_nl = FreeFlyer(backend='cs')
if True:
    # gradient-based optimization with casadi
    ocp = cs.Opti()
    x_vars = ocp.variable(13, N)
    u_vars = ocp.variable(6, N)
    dt_vars = ocp.variable(1)
    delta = ocp.variable(1)

    ocp.set_initial(x_vars, x_uw_fbl_sp.T)
    ocp.set_initial(u_vars, u_uw_fbl_sp.T)
    ocp.set_initial(dt_vars, 0.02)

    ocp.subject_to(dt_vars >= 0)
    ocp.subject_to(delta >= 0)

    for i in range(N):
        # ocp.subject_to(u_fbl_cs(x_vars[:,i],u_vars[:,i])[0:2] >= alpha*(uw_robot.U.lower_bounds[0:2]))
        # ocp.subject_to(u_fbl_cs(x_vars[:,i],u_vars[:,i])[0:2] <= alpha*(uw_robot.U.upper_bounds[0:2]))
        ocp.subject_to(u_vars[:,i] >= alpha * uw_robot.U.lower_bounds)
        ocp.subject_to(u_vars[:,i] <= alpha * uw_robot.U.upper_bounds)
        # ocp.subject_to(u_vars[:,i] >= alpha * u_fbl_cs(x_vars[:,i],uw_robot.U.lower_bounds))
        # ocp.subject_to(u_vars[:,i] <= alpha * u_fbl_cs(x_vars[:,i],uw_robot.U.upper_bounds))

    # constraints
    for i in range(N-1):
        ocp.subject_to(x_vars[:,i+1] == uw_robot.step(x_vars[:,i], u_vars[:,i], dt_vars))
    
    # enforce position equality with the free flyer
    n_equal = 7 # first n_equal states 
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
    data = np.load("solutions/uw_solution.npz")
    x_uw = data['x_uw']
    u_uw = data['u_uw']
    dt_uw = data['dt_uw']

# t_uw = np.linspace(0, dt_uw*N, N)
# u_uw_fbl = np.array([u_fbl(x_uw[i,:],u_uw[i,:])[0:2].squeeze() for i in range(N)])
# u_max = np.max(np.abs(u_uw_fbl), axis=0)
# print(f"\nAlpha for replanning uw to have same alpha: {np.max([u_max/(uw_robot.U.upper_bounds[0:2]) for i in range(N)])}")
# print(f"This should be the same as the solution to Prob 1")
# print(f"Replanned dt_uw: {dt_uw} (was {dt}) which is {(dt_uw)/dt*100}%")

# plot the trajectory
plot_planning_results(uw_robot, t_uw, x_uw, u_uw, X0, Xf, [XA], Obs, alpha=alpha,
                      path="stl_mapping/Planning/figures/uw_trajectory.png")
