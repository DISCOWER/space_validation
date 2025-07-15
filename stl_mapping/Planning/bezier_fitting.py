import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from Utilities.beziers import eval_bezier

data = np.load('Planning/solutions/sp_solution_quat.npz')

x_ff = data['x_ff']
u_ff = data['u_ff']
alpha = data['alpha']
dt = data['dt']

N = x_ff.shape[0]
n = 3
times = np.arange(N) * dt

# create d-th order Bezier curve from variables
d = 3

constraints = []
cost = 0

r_vars = [cp.Variable((n, d+1)) for _ in range(N-1)]
dr_vars = [cp.Variable((n, d)) for _ in range(N-1)]
ddr_vars = [cp.Variable((n, d-1)) for _ in range(N-1)]
for i in range(N-1):
    for c in range(d):
        constraints += [dr_vars[i][:,c] == (d)*(r_vars[i][:,c+1] - r_vars[i][:,c])/dt]
    for c in range(d-1):
        constraints += [ddr_vars[i][:,c] == (d-1)*(dr_vars[i][:,c+1] - dr_vars[i][:,c])/dt]

# end-point position constraints
for i in range(N-1):
    constraints += [r_vars[i][:,0] == x_ff[i,0:3]]
    constraints += [r_vars[i][:,-1] == x_ff[i+1,0:3]]

# end-point velocity constraints
for i in range(N-1):
    constraints += [dr_vars[i][:,0] == x_ff[i,7:10]]
    constraints += [dr_vars[i][:,-1] == x_ff[i+1,7:10]]

# cost
for i in range(N-1):
    cost += cp.sum_squares(ddr_vars[i])

prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver=cp.SCS)

# Gather the results and save
r_sols = [r_vars[i].value for i in range(N-1)]
dr_sols = [dr_vars[i].value for i in range(N-1)]
ddr_sols = [ddr_vars[i].value for i in range(N-1)]
r_sols, dr_sols, ddr_sols = np.array(r_sols), np.array(dr_sols), np.array(ddr_sols)

q_sols = [np.array([x_ff[i, 3:7], x_ff[i+1, 3:7]]) for i in range(N-1)]
q_sols = np.array(q_sols)

np.savez('Planning/solutions/sp_solution_bezier.npz',
         r=r_sols, dr=dr_sols, ddr=ddr_sols, q=q_sols, dt=dt, alpha=alpha, times=times)


r_vals = [eval_bezier(r_vars[i].value) for i in range(N-1)]
dr_vals = [eval_bezier(dr_vars[i].value) for i in range(N-1)]
ddr_vals = [eval_bezier(ddr_vars[i].value) for i in range(N-1)]

# Print the results
print(f"Optimal cost: {prob.value}")

# Plot the results
fig, axs = plt.subplots(1,3, figsize=(15, 10))

axs[0].plot(x_ff[:, 0], x_ff[:, 1], 'o', label='Trajectory')
for i in range(N-1):
    axs[0].plot(r_vals[i][0, :], r_vals[i][1, :])
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_title('Trajectory and Bezier Curve')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].plot(times, x_ff[:,7], 'bo', label='Velocity')
axs[1].plot(times, x_ff[:,8], 'ro')
for i in range(N-1):
    i_time = np.linspace(i*dt, (i+1)*dt, r_vals[i].shape[1])
    axs[1].plot(i_time, dr_vals[i][0, :], 'b', label=f'Bezier {i}')
    axs[1].plot(i_time, dr_vals[i][1, :], 'r')
    # plot the control points
    i_time = np.linspace(i*dt, (i+1)*dt, dr_vars[i].shape[1])
    axs[1].plot(i_time, dr_vars[i].value[0, :], 'ko')
    axs[1].plot(i_time, dr_vars[i].value[1, :], 'ko')
axs[1].set_title('Velocity and Bezier Curve')
axs[1].set_xlabel('vx')
axs[1].set_ylabel('vy')

for i in range(N-1):
    i_time = np.linspace(i*dt, (i+1)*dt, r_vals[i].shape[1])
    axs[2].plot(i_time, ddr_vals[i][0, :], 'b', label=f'Bezier {i}')
    axs[2].plot(i_time, ddr_vals[i][1, :], 'r')
    # plot the control points
    i_time = np.linspace(i*dt, (i+1)*dt, ddr_vars[i].shape[1])
    axs[2].plot(i_time, ddr_vars[i].value[0, :], 'ko')
    axs[2].plot(i_time, ddr_vars[i].value[1, :], 'ko')
axs[2].set_title('Acceleration Bezier Curve')
axs[2].set_xlabel('ax')
axs[2].set_ylabel('ay')
plt.show()

