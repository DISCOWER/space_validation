import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from Utilities.beziers import eval_bezier

def solve_position(x):
    N = x.shape[0]
    n = 3
    d = 3
    r_vars = [cp.Variable((n, d+1)) for _ in range(N-1)]
    dr_vars = [cp.Variable((n, d)) for _ in range(N-1)]
    ddr_vars = [cp.Variable((n, d-1)) for _ in range(N-1)]

    constraints = []
    # control point constraints
    for i in range(N-1):
        for c in range(d):
            constraints += [dr_vars[i][:,c] == (d)*(r_vars[i][:,c+1] - r_vars[i][:,c])]
        for c in range(d-1):
            constraints += [ddr_vars[i][:,c] == (d-1)*(dr_vars[i][:,c+1] - dr_vars[i][:,c])]
    # continuity constraints
    for i in range(N-2):
        constraints += [r_vars[i][:,-1] == r_vars[i+1][:,0]]
        constraints += [dr_vars[i][:,-1] == dr_vars[i+1][:,0]]
        constraints += [ddr_vars[i][:,-1] == ddr_vars[i+1][:,0]]
    # end-point position constraints
    delta_r, delta_dr = cp.Variable(1), cp.Variable(1)
    constraints += [delta_r >= 0, delta_dr >= 0]
    add_slack = True
    for i in range(N-1):
        if add_slack:
            constraints += [r_vars[i][:,0] <= x[i,0:3] + delta_r]
            constraints += [r_vars[i][:,0] >= x[i,0:3] - delta_r]
            constraints += [r_vars[i][:,-1] <= x[i+1,0:3] + delta_r]
            constraints += [r_vars[i][:,-1] >= x[i+1,0:3] - delta_r]
        else:
            constraints += [r_vars[i][:,0] == x[i,0:3]]
            constraints += [r_vars[i][:,-1] == x[i+1,0:3]]
    # cost
    cost = 0
    for i in range(N-1):
        cost += cp.sum_squares(ddr_vars[i])
    if add_slack:
        cost += 1000*delta_r
    # solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    # Gather results
    r_sols = [r_vars[i].value for i in range(N-1)]
    dr_sols = [dr_vars[i].value for i in range(N-1)]
    ddr_sols = [ddr_vars[i].value for i in range(N-1)]
    r_sols, dr_sols, ddr_sols = np.array(r_sols), np.array(dr_sols), np.array(ddr_sols)
    return r_vars, r_sols

def solve_velocity(x):
    N = x.shape[0]
    n = 3
    d = 4
    dr_vars = [cp.Variable((n, d)) for _ in range(N-1)]
    ddr_vars = [cp.Variable((n, d-1)) for _ in range(N-1)]
    dddr_vars = [cp.Variable((n, d-2)) for _ in range(N-1)]

    constraints = []
    # control point constraints
    for i in range(N-1):
        for c in range(d-1):
            constraints += [ddr_vars[i][:,c] == (d-1)*(dr_vars[i][:,c+1] - dr_vars[i][:,c])]
        for c in range(d-2):
            constraints += [dddr_vars[i][:,c] == (d-2)*(ddr_vars[i][:,c+1] - ddr_vars[i][:,c])]
    # continuity constraints
    for i in range(N-2):
        constraints += [dr_vars[i][:,-1] == dr_vars[i+1][:,0]]
        constraints += [ddr_vars[i][:,-1] == ddr_vars[i+1][:,0]]
    # end-point velocity constraints
    delta_r, delta_dr = cp.Variable(1), cp.Variable(1)
    constraints += [delta_r >= 0, delta_dr >= 0]
    add_slack = True
    for i in range(N-1):
        if add_slack:
            constraints += [dr_vars[i][:,0] <= x[i,7:10] + delta_dr]
            constraints += [dr_vars[i][:,0] >= x[i,7:10] - delta_dr]
            constraints += [dr_vars[i][:,-1] <= x[i+1,7:10] + delta_dr]
            constraints += [dr_vars[i][:,-1] >= x[i+1,7:10] - delta_dr]
        else:
            constraints += [dr_vars[i][:,0] == x[i,7:10]]
            constraints += [dr_vars[i][:,-1] == x[i+1,7:10]]
    # cost
    cost = 0
    for i in range(N-1):
        cost += cp.sum_squares(dddr_vars[i])
    if add_slack:
        cost += 1000*delta_dr
    # solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    # Gather results
    dr_sols = [dr_vars[i].value for i in range(N-1)]
    ddr_sols = [ddr_vars[i].value for i in range(N-1)]
    dr_sols, ddr_sols = np.array(dr_sols), np.array(ddr_sols)
    return dr_vars, dr_sols


def bezier_fitting(data, model_name):
    x = data['x']
    u = data['u']
    alpha = data['alpha']
    dt = data['dt']
    times = data['times']

    N = x.shape[0]

    r_vars, r_sols = solve_position(x)
    dr_vars, dr_sols = solve_velocity(x)

    # Gather the results and save
    r_sols, dr_sols = np.array(r_sols), np.array(dr_sols)

    q_sols = [np.array([x[i, 3:7], x[i+1, 3:7]]) for i in range(N-1)]
    q_sols = np.array(q_sols)
    # print(f"q_sols: {q_sols}")

    np.savez(f'stl_mapping/Planning/solutions/{model_name}_solution_bezier.npz',
            r=r_sols, dr=dr_sols, q=q_sols, dt=dt, alpha=alpha, times=times)


    r_vals = [eval_bezier(r_vars[i].value) for i in range(N-1)]
    dr_vals = [eval_bezier(dr_vars[i].value) for i in range(N-1)]

    # Print the results
    # print(f"Optimal cost: {prob.value}")

    # Plot the results
    fig, axs = plt.subplots(1,4, figsize=(20, 10))

    axs[0].plot(x[:, 0], x[:, 1], 'o', label='Trajectory')
    for i in range(N-1):
        axs[0].plot(r_vals[i][0, :], r_vals[i][1, :])
        # plot an arrow of the velocity direction
        axs[0].quiver(r_vals[i][0, -1], r_vals[i][1, -1], x[i, 7], x[i, 8], color='r', scale=1)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_title('Trajectory and Bezier Curve')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    for i in range(N-1):
        i_time = np.linspace(i*dt, (i+1)*dt, r_vals[i].shape[1])
        axs[1].plot(i_time, r_vals[i][0, :], 'b', label=f'Bezier {i}')
        axs[1].plot(i_time, r_vals[i][1, :], 'r')
        i_time = np.linspace(i*dt, (i+1)*dt, r_vars[i].shape[1])
        axs[1].plot(i_time, r_vars[i].value[0, :], 'ko')
        axs[1].plot(i_time, r_vars[i].value[1, :], 'ko')
    axs[1].set_title('Position over time')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('x/y')


    for i in range(N-1):
        i_time = np.linspace(i*dt, (i+1)*dt, r_vals[i].shape[1])
        axs[2].plot(i_time, dr_vals[i][0, :], 'b', label=f'Bezier {i}')
        axs[2].plot(i_time, dr_vals[i][1, :], 'r')
        # plot the control points
        i_time = np.linspace(i*dt, (i+1)*dt, dr_vars[i].shape[1])
        axs[2].plot(i_time, dr_vars[i].value[0, :], 'ko')
        axs[2].plot(i_time, dr_vars[i].value[1, :], 'ko')
        # plot the reference trajectory
        axs[2].plot(times,x[:, 7], 'bo--', label='Reference vx')
        axs[2].plot(times,x[:, 8], 'ro--', label='Reference vy')
    axs[2].set_title('Velocity and Bezier Curve')
    axs[2].set_xlabel('vx')
    axs[2].set_ylabel('vy')

    # for i in range(N-1):
    #     i_time = np.linspace(i*dt, (i+1)*dt, r_vals[i].shape[1])
    #     axs[3].plot(i_time, ddr_vals[i][0, :], 'b', label=f'Bezier {i}')
    #     axs[3].plot(i_time, ddr_vals[i][1, :], 'r')
    #     # plot the control points
    #     i_time = np.linspace(i*dt, (i+1)*dt, ddr_vars[i].shape[1])
    #     axs[3].plot(i_time, ddr_vars[i].value[0, :], 'ko')
    #     axs[3].plot(i_time, ddr_vars[i].value[1, :], 'ko')
    # axs[3].set_title('Acceleration Bezier Curve')
    # axs[3].set_xlabel('ax')
    # axs[3].set_ylabel('ay')
    plt.savefig(f'stl_mapping/Planning/figures/{model_name}_solution_bezier.png',dpi=300)



if __name__ == "__main__":
    model_name = 'bluerov' # Change to 'atmos' or 'bluerov if needed
    data = np.load(f'stl_mapping/Planning/solutions/{model_name}_nonlinear_solution.npz')
    bezier_fitting(data, model_name)
