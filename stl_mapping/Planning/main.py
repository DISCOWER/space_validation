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
from Utilities.plotting.plotting import plot_planning_results
from Utilities.smarc_modelling.src.smarc_modelling.vehicles.BlueROV import BlueROV 
from Planning.bezier_fitting import bezier_fitting

# hyperparameters
N = 50          # number of time steps
dt = 1.5        # time step size
t0 = 0          # initial time
tf = (N-1)*dt   # final time

bigM = 1e4

# Robot
sp_robot = LinearFreeFlyer6DoF()

euler_order = 'xyz'

### STL Specification
scenario = 'toy-example'
# scenario = 'paper3D'

if scenario == 'toy-example':
    D3 = False
    depth = 0.0
    X0 = HyperRectangle(center=np.array([0.75, 0, depth]),      size=np.array([0.5, 0.5, 0.5]))
    Xf = HyperRectangle(center=np.array([2.75, 1.25, depth]),   size=np.array([0.5, 0.5, 0.5]))
    XA = HyperRectangle(center=np.array([2.5, -1.25, depth,  -np.pi/2]), size=np.array([0.5, 0.5, 0.5,  np.pi/4]))
    XB = HyperRectangle(center=np.array([1.5, 0.75, depth,  np.pi/2]),  size=np.array([0.5, 0.5, 0.5,  np.pi/4]))
    # XA = HyperRectangle(np.array([2.5, -1.0, depth,  -np.pi/2-np.pi/8, -np.pi/2-np.pi/8]), 
    #                     np.array([2.6, -0.9, depth+0.1,  -np.pi/2+np.pi/8, -np.pi/2+np.pi/8]))
    # later converted to polytopes for predicates of the form Ax \leq b: Polytope = (A,b)
    # XA = HyperRectangle(np.array([2.5, -1.0, depth]), np.array([2.6, -0.9, depth+0.1]))

    Obs = []
    RoIs = [XA, XB]

    World = HyperRectangle(np.array([0, -1.5, -10, -10]), np.array([4, 1.5, 10, 10]))
    phi = Pred("AND", preds=[
        Pred("G", [t0,t0], preds=[Pred("MU", preds=[Polytope(X0)], dims=[0,1,2])]),
        Pred("G", [tf,tf], preds=[Pred("MU", preds=[Polytope(Xf)], dims=[0,1,2])]),
        Pred("F", [t0,tf/2], preds=[Pred("MU", preds=[Polytope(XA)], dims=[0,1,2, 5])]),
        Pred("F", [tf/2,tf], preds=[Pred("MU", preds=[Polytope(XB)], dims=[0,1,2, 5])]),
        Pred("G", [t0,tf], preds=[Pred("MU", preds=[Polytope(World)], dims=[0,1,6,7])]),
    ])
    spec = Spec(phi, t0, tf)
elif scenario == 'paper3D':
    D3 = True
    X0 = HyperRectangle(center=np.array([1.0, 0, 1.5]), size=np.array([0.5, 0.5, 0.5]))
    Xf = HyperRectangle(center=np.array([1.0, 0, 1.5]), size=np.array([0.5, 0.5, 0.5]))

    # Three observation tasks, each having two possible views around the Obstacle.
    Obs1 = HyperRectangle(center=np.array([4.0, 0, 1.5]),                   size=np.array([1.0, 1.5, 0.5]))
    XA1 = HyperRectangle(center=np.array([3.0, 0, 1.5, 0, 0]),              size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
    XA2 = HyperRectangle(center=np.array([5.0, 0, 1.5, 0, -np.pi]),         size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
    XB1 = HyperRectangle(center=np.array([4.0, -1.25, 1.5, 0, np.pi/2]),    size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
    XB2 = HyperRectangle(center=np.array([4.0, 1.25, 1.5, 0, -np.pi/2]),    size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
    XC1 = HyperRectangle(center=np.array([4.0, 0, 2.25, np.pi/2, 0]),      size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
    XC2 = HyperRectangle(center=np.array([4.0, 0, 0.75, -np.pi/2, 0]),       size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))

    Obs = [Obs1]
    RoIs = [XA2, XB1, XB2, XC1, XC2]

    phi = Pred("AND", preds=[
        Pred("G", [t0,t0], preds=[Pred("MU", preds=[Polytope(X0)], dims=[0,1,2])]),
        Pred("G", [tf,tf], preds=[Pred("MU", preds=[Polytope(Xf)], dims=[0,1,2])]),
        Pred("G", [t0,tf], preds= [Pred("NEG", preds= [Pred("MU", preds=[Polytope(Obs1)], dims=[0,1,2])] )] ),
        Pred("OR", preds=[
            # Pred("G", [10,12], preds=[Pred("MU", preds=[Polytope(XA1)], dims=[0,1,2, 4,5])]),
            Pred("G", [20,25], preds=[Pred("MU", preds=[Polytope(XA2)], dims=[0,1,2, 4,5])])
        ]),
        Pred("OR", preds=[
            # Pred("F", [30,35], preds=[Pred("MU", preds=[Polytope(XB1)], dims=[0,1,2, 4,5])]),
            Pred("F", [30,35], preds=[Pred("MU", preds=[Polytope(XB2)], dims=[0,1,2, 4,5])])
        ]),
        Pred("OR", preds=[
            # Pred("F", [35,40], preds=[Pred("MU", preds=[Polytope(XC1)], dims=[0,1,2, 4,5])]),
            Pred("F", [35,40], preds=[Pred("MU", preds=[Polytope(XC2)], dims=[0,1,2, 4,5])])
        ])
    ])
    spec = Spec(phi, t0, tf)

# Initial Motion planner on Linear Model
if False:
    opt = gp.Model("prob1")
    x_vars = opt.addMVar((N, sp_robot.n_x), lb=-np.inf, ub=np.inf, name="X")
    u_vars = opt.addMVar((N-1, sp_robot.n_u), lb=-np.inf, ub=np.inf, name="U")
    t_sp = np.linspace(0,(N-1)*dt, N)
    # alpha_vars = opt.addVar(lb=0, ub=1, name="alpha")
    alpha_vars = opt.addVar(lb=1, ub=100, name="alpha")
    opt.update()

    items = OptProbItems(x_vars, u_vars, t_sp)

    for i in range(N-1):
        # opt.addConstrs((u_vars[i, j] >= alpha_vars*sp_robot.U_effective.lower_bounds[j] for j in range(sp_robot.n_u)))
        # opt.addConstrs((u_vars[i, j] <= alpha_vars*sp_robot.U_effective.upper_bounds[j] for j in range(sp_robot.n_u)))
        try:
            opt.addConstrs((u_vars[i, j] >= sp_robot.calculate_U_effective(np.zeros((13,)), alpha_vars).lower_bounds[j] for j in range(sp_robot.n_u)))
            opt.addConstrs((u_vars[i, j] <= sp_robot.calculate_U_effective(np.zeros((13,)), alpha_vars).upper_bounds[j] for j in range(sp_robot.n_u)))
            # opt.addConstrs((u_vars[i, j] >= sp_robot.U.lower_bounds[j] - alpha_vars*sum(abs(sp_robot.K[j,:])*sp_robot.D.lower_bounds) for j in range(sp_robot.n_u)))
            # opt.addConstrs((u_vars[i, j] <= sp_robot.U.upper_bounds[j] + alpha_vars*sum(abs(sp_robot.K[j,:])*sp_robot.D.upper_bounds) for j in range(sp_robot.n_u)))
        except Exception as e:
            print(f"Error adding constraints for u_vars at time step {i}: {e}")

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
    act_absu_scaled_vars = opt.addMVar((N-1, sp_robot.n_u), lb=0, ub=np.inf)
    act_absu_var = opt.addVar(lb=0, ub=np.inf)
    Q = np.array([1., 1., 1., 10., 10., 10.])
    opt.addConstrs((act_absu_vars[i, j] == gp.abs_(u_vars[i, j]) for i in range(N-1) for j in range(sp_robot.n_u)), name="act_abs")
    opt.addConstrs((act_absu_scaled_vars[i, j] == Q[j]*act_absu_vars[i, j] for i in range(N-1) for j in range(sp_robot.n_u)), name="act_abs_scaled")
    opt.addConstr(act_absu_var == gp.quicksum([act_absu_scaled_vars[i, j] for i in range(N-1) for j in range(sp_robot.n_u)]), name="act_abs_sum")

    # Final cost
    opt.addConstr(cost_var == -1*alpha_vars - 10000*spec.phi.rho + 0.01*act_absu_var)# - act_quadu_var)
    opt.setObjective(cost_var, gp.GRB.MINIMIZE)
    opt.setParam('OutputFlag', 0)  # Suppress Gurobi output
    opt.optimize()
    if opt.status == gp.GRB.OPTIMAL:
        print(f"Optimal solution found")
        print(f"\nAlpha from solving Prob 1: {alpha_vars.X}")
        print(f"This is how much we can scale D while still satisfying phi")
        print(f"Spatial robustness: rho = {spec.phi.rho.X}")
        # print(f"Control integral summed: {act_quadu_var.X}")
    else:
        print(f"No optimal solution found")
        print(f"Status: {scs[opt.status]}")

    x_ff = x_vars.X
    u_ff = u_vars.X
    alpha = alpha_vars.X
    # save x_ff and u_ff to a csv file
    np.savez('stl_mapping/Planning/solutions/atmos_linear_solution.npz', x=x_ff, u=u_ff, dt=dt, alpha=alpha, times=t_sp)
else:
    # or load instead
    data = np.load('stl_mapping/Planning/solutions/atmos_linear_solution.npz')
    x_ff = data['x']
    u_ff = data['u']
    alpha = float(data['alpha'])
    dt = data['dt']
    t_sp = data['times']
    print(f"alpha from linear STL planner: {alpha}")
    

plot_planning_results(sp_robot, t_sp, x_ff, u_ff, X0, Xf, RoIs, Obs, alpha=alpha, D3=D3, plot=True,
                      path="stl_mapping/Planning/figures/atmos_linear_trajectory_euler.png")

np.savez('stl_mapping/Planning/solutions/atmos_linear_solution_euler_inertial.npz', x=x_ff, u=u_ff, dt=dt, alpha=alpha, times=t_sp)
# np.savetxt("x_ff.csv", x_ff, delimiter=',', fmt="%.4f")

# exit()

#! Convert [p:ENU, e:FLU->ENU, v:ENU, w:FLU] to [p:ENU, q:FLU->ENU, v:FLU, w:FLU]
x_ff_converted = np.zeros((x_ff.shape[0], x_ff.shape[1] + 1))
for i in range(x_ff.shape[0]):
    x_ff_converted[i,0:3] = x_ff[i,0:3]
    x_ff_converted[i,3:7] = euler_to_quat_np(x_ff[i, 3:6], order=euler_order)
    q = R.from_quat(x_ff_converted[i, 3:7], scalar_first=True)
    x_ff_converted[i,7:10] = q.inv().apply(x_ff[i, 6:9])
    # x_ff_converted[i,7:10] = x_ff[i, 6:9]
    x_ff_converted[i,10:13] = x_ff[i, 9:12] #enu_to_flu(x_ff[i, 9:12], x_ff_converted[i, 3:7])
x_ff = x_ff_converted

#! Convert [F: ENU, tau: FLU] to [F: FLU, tau:FLU]
u_ff_converted = np.zeros_like(u_ff)
for i in range(u_ff.shape[0]):
    q = R.from_quat(x_ff[i, 3:7], scalar_first=True)
    u_ff_converted[i, :] = np.hstack((q.inv().apply(u_ff[i, 0:3]),# + sp_robot.mass*np.cross(x_ff_converted[i, 10:13], x_ff_converted[i, 7:10]), 
                                      q.inv().apply(u_ff[i, 3:6])))
    # u_ff_converted[i, :] = u_ff[i, :]  # keep the angular velocities in FLU
u_ff = u_ff_converted

# plot the trajectory
plot_planning_results(sp_robot, t_sp, x_ff, u_ff, X0, Xf, RoIs, Obs, alpha=alpha, D3=D3,
                      path="stl_mapping/Planning/figures/atmos_linear_trajectory.png")

np.savez('stl_mapping/Planning/solutions/atmos_linear_solution_quat_body.npz', x=x_ff, u=u_ff, dt=dt, alpha=alpha, times=t_sp)

#TODO: this is a valid conversion (Lin to non-lin) if the system (with zero roll) is differentially flat?
#TODO: this means that this linear decoupling is valid, and x_ff and u_ff are valid for the nonlinear space robot
#TODO: CHECK THIS!
sp_robot_nl = FreeFlyer()
# x_ff_nl = np.zeros((x_ff.shape[0], x_ff.shape[1]))
# u_ff_nl = u_ff
# x_ff_i = copy.deepcopy(x_ff[0, :])
# x_ff_nl[0, :] = x_ff_i
# for i in range(x_ff.shape[0]-1):
#     # q = R.from_quat(x_ff_nl[i, 3:7], scalar_first=True)
#     u_ff_i = u_ff[i, :]  # convert u from body FLU to ENU frame
#     x_ff_i = sp_robot_nl.step(x_ff_i, u_ff_i, dt)
#     x_ff_i[3:7] /= np.linalg.norm(x_ff_i[3:7])  # normalize quaternion
#     x_ff_nl[i+1, :] = x_ff_i
# plot_planning_results(sp_robot_nl, t_sp, x_ff_nl, u_ff, X0, Xf, RoIs, Obs, alpha=alpha,
#                       path="stl_mapping/Planning/figures/atmos_nonlinear_trajectory.png")


if True:
    ocp = cs.Opti()
    x_vars = ocp.variable(13, N)
    u_vars = ocp.variable(6, N-1)
    delta = ocp.variable(1)
    alpha_var = ocp.variable(1)

    ocp.set_initial(x_vars, x_ff.T)
    ocp.set_initial(u_vars, u_ff.T)
    ocp.set_initial(alpha_var, alpha)
    ocp.set_initial(delta, 0)

    ocp.subject_to(delta >= 0)

    for i in range(N-1):
        # ocp.subject_to(u_vars[:,i] >= sp_robot_nl.U.lower_bounds)
        # ocp.subject_to(u_vars[:,i] <= sp_robot_nl.U.upper_bounds)
        ocp.subject_to(u_vars[:,i] >= sp_robot_nl.calculate_U_effective(np.zeros((13,1)),alpha-0.5).lower_bounds)
        ocp.subject_to(u_vars[:,i] <= sp_robot_nl.calculate_U_effective(np.zeros((13,1)),alpha-0.5).upper_bounds)

    for i in range(N-1):
        ocp.subject_to(x_vars[:,i+1] == sp_robot_nl.step(x_vars[:,i], u_vars[:,i], dt, normalize=True))

    n_equal = 7
    for i in range(N):
        ocp.subject_to(x_vars[0:n_equal,i] <= x_ff[i,0:n_equal] + delta*np.ones(n_equal))
        ocp.subject_to(x_vars[0:n_equal,i] >= x_ff[i,0:n_equal] - delta*np.ones(n_equal))
    ocp.subject_to(x_vars[:7,0] == x_ff[0,:7])
    ocp.subject_to(x_vars[:7,N-1] == x_ff[N-1,:7])


    quad_u_var = 0
    Q = np.diag([1., 1., 1., 10., 10., 10.])
    for i in range(N-1):
        quad_u_var += cs.mtimes(u_vars[:,i].T, Q @ u_vars[:,i])
    ocp.minimize(quad_u_var + 1e8*delta)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'verbose':False, 'ipopt.tol': 1e-4, 'ipopt.max_iter': 1000}
    ocp.solver('ipopt',opts)

    try:
        print(f"\n\nSolving nl atmos optimization problem...")
        sol = ocp.solve()
        x_ff_nl = sol.value(x_vars).T
        u_ff_nl = sol.value(u_vars).T
        t_ff_nl = np.linspace(0, (N-1)*dt, N)
        print(f"delta: {sol.value(delta)}")
    except Exception as e:
        print(f"Solution not found: {e}")

plot_planning_results(sp_robot_nl, t_ff_nl, x_ff_nl, u_ff_nl, X0, Xf, RoIs, Obs, alpha=alpha, D3=D3,
                      path="stl_mapping/Planning/figures/atmos_opt_nonlinear_trajectory.png")

np.savez('stl_mapping/Planning/solutions/atmos_nonlinear_solution.npz', x=x_ff_nl, u=u_ff, dt=dt, alpha=alpha, times=t_sp)
data = np.load('stl_mapping/Planning/solutions/atmos_nonlinear_solution.npz')
bezier_fitting(data,'atmos')

#TODO: Feedback linearization test
#TODO: we test feedback linearization controller for the underwater robot to behave like a free flyer
uw_robot = BlueROV()
sp_robot_nl = FreeFlyer()

def u_fbl(x, u):
    '''
    Feedback linearization control for the underwater robot
    Args:
        x: state of FF [p: ENU, q:FLU->ENU, v:ENU , w:FLU] in ENU frame
        v: control input of FF in ENU frame
    Returns:
        u_fbl: control input for the space robot in ENU body frame
    '''
    fx_sp = sp_robot_nl.calculate_fx(x)
    gx_sp = sp_robot_nl.calculate_gx(x)
    dx_sp = fx_sp + gx_sp@u

    fx_uw = uw_robot.calculate_fx(x)
    gx_uw = uw_robot.calculate_gx(x)
    u_fbl = np.linalg.pinv(gx_uw)@(dx_sp - fx_uw)

    return u_fbl


x_ff_i = copy.deepcopy(x_ff_nl[0, :])

x_uw_fbl_sp = np.zeros((N, sp_robot_nl.n_x))
x_uw_fbl_sp[0, :] = x_ff_i
u_uw_fbl_sp = np.zeros((N, sp_robot_nl.n_u))
for i in range(N-1):
    u_uw = u_fbl(x_ff_i, u_ff_nl[i, :])
    x_ff_i = uw_robot.step(x_ff_i, u_uw, dt)
    x_ff_i[3:7] /= np.linalg.norm(x_ff_i[3:7])

    # save state and control
    x_uw_fbl_sp[i+1, :] = x_ff_i
    u_uw_fbl_sp[i, :] = u_uw


# save the trajectory
np.savez('stl_mapping/Planning/solutions/bluerov_nonlinear_solution_fbl.npz', x=x_uw_fbl_sp, u=u_uw_fbl_sp, dt=dt, alpha=alpha, times=t_sp)

# plot the trajectory
plot_planning_results(uw_robot, t_sp, x_uw_fbl_sp, u_uw_fbl_sp, X0, Xf, RoIs, Obs, alpha=alpha, D3=D3,
                      path="stl_mapping/Planning/figures/bluerov_nonlinear_trajectory_fbl.png")

# Analysis of alpha
u_max = np.max(np.abs(u_uw_fbl_sp), axis=0)
print(f"\nAlpha if u_fbl(x_sp,u_sp) applied to BlueROV directly: {np.max(u_max/uw_robot.U.upper_bounds)} (was {alpha} for ff)")
print(f"This is lower because BlueROV is faster than free flyer")
print(f"U should become {np.max(u_max)/alpha} for same alpha (was {uw_robot.U.upper_bounds[0]}) which is {(np.max(u_max)/alpha)/uw_robot.U.upper_bounds[0]*100}%")
# exit()

#TODO: now we either solve the time-scaling of trajectory and control input
#TODO: or we solve the optimization problem to make alpha equal, we need to figure out what is warranted
#TODO: - time-scaling?
#TODO: - control bound scaling? (only valid for faster systems, but still, is it valid?)
#TODO: - trajectory optimization? (this should be valid, but perhaps dirtier than warranted)
uw_robot = BlueROV(iX=cs.MX)
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

    ocp.subject_to(0.1 <= dt_vars+0.2)
    ocp.subject_to(dt_vars <= dt)
    ocp.subject_to(delta >= 0)

    for i in range(N):
        ocp.subject_to(u_vars[:,i] >= uw_robot.calculate_U_effective(np.zeros((13,1)),alpha).lower_bounds)
        ocp.subject_to(u_vars[:,i] <= uw_robot.calculate_U_effective(np.zeros((13,1)),alpha).upper_bounds)

        # ocp.subject_to(u_vars[:,i] >= alpha * uw_robot.calculate_U_effective(x_vars[:,i]).lower_bounds)
        # ocp.subject_to(u_vars[:,i] <= alpha * uw_robot.calculate_U_effective(x_vars[:,i]).upper_bounds)
        # for j in range(6)
        # for j in range(6):
        #     ocp.subject_to(u_vars[j,i] >= uw_robot.U.lower_bounds[j] + alpha * uw_robot.calculate_sum_K_D(x_vars[:,i], uw_robot.D.lower_bounds)[j])
        #     ocp.subject_to(u_vars[j,i] <= uw_robot.U.upper_bounds[j] + alpha * uw_robot.calculate_sum_K_D(x_vars[:,i], uw_robot.D.upper_bounds)[j])

    # constraints
    for i in range(N-1):
        ocp.subject_to(x_vars[:,i+1] == uw_robot.step(x_vars[:,i], u_vars[:,i], dt_vars, normalize=True))

    # enforce position equality with the free flyer
    n_equal = 7 # first n_equal states: [p,q]
    for i in range(N):
        # ocp.subject_to(x_vars[0:n_equal,i] == x_uw_fbl_sp[i,0:n_equal])
        ocp.subject_to(x_vars[0:n_equal,i] <= x_ff[i,0:n_equal] + delta * np.ones(n_equal))
        ocp.subject_to(x_vars[0:n_equal,i] >= x_ff[i,0:n_equal] - delta * np.ones(n_equal))
    ocp.subject_to(x_vars[7:,0] == x_ff[0,7:])
    ocp.subject_to(x_vars[7:,-1] == x_ff[-1,7:])

    # objective
    quad_u_var = 0
    Q = np.array([1., 1., 1., 10., 10., 10.])
    for i in range(N):
        quad_u_var += cs.mtimes(u_vars[:,i].T, Q * u_vars[:,i])
    cost_var = 100*dt_vars + 1e3 * delta + 1e-4 * quad_u_var
    ocp.minimize(cost_var)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'verbose':False, 'ipopt.tol': 1e-4, 'ipopt.max_iter': 1000}
    ocp.solver('ipopt',opts)

    try:
        print(f"\n\nSolving nl bluerov optimization problem...")
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

# plot the trajectory
plot_planning_results(uw_robot, t_uw, x_uw, u_uw, X0, Xf, RoIs, Obs, alpha=alpha, D3=D3,
                      path="stl_mapping/Planning/figures/bluerov_nonlinear_trajectory.png")

# save the trajectory
np.savez('stl_mapping/Planning/solutions/bluerov_nonlinear_solution.npz', x=x_uw, u=u_uw, dt=dt_uw, alpha=alpha, times=t_uw)
data = np.load('stl_mapping/Planning/solutions/bluerov_nonlinear_solution.npz')
bezier_fitting(data, 'bluerov')