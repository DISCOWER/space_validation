import numpy as np
import matplotlib.pyplot as plt
from Utilities.rotations import quat_to_euler_np, euler_to_quat_np

def plot_planning_results(robot, t, x, u, 
                          X0=None, Xf=None, ROIs=None, Obs=None, alpha=1.0,
                          in_axs = None, plot=True,
                          path="stl_mapping/Planning/figures/sp_trajectory.png"):
    if x.shape[1] == 12:
        p, e, v, w = x[:, 0:3], x[:, 3:6], x[:, 6:9], x[:, 9:12]
        q = np.zeros((x.shape[0], 4))
        for i in range(x.shape[0]):
            q[i, :] = euler_to_quat_np(e[i, :], order='zyx')
        x = np.hstack((p, q, v, w))
        print(f"Converted state representation:\n{x}")
    elif x.shape[1] == 13:
        p, q, v, w = x[:, 0:3], x[:, 3:7], x[:, 7:10], x[:, 10:13]
    else:
        raise ValueError("State vector x must have 12 or 13 columns.")
    

    # plot the trajectory
    if in_axs is None:
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(2,4, figure=fig)
        ax_p = fig.add_subplot(gs[:,0])
        ax_v = fig.add_subplot(gs[:,1])
        ax_q = fig.add_subplot(gs[0,2])
        ax_w = fig.add_subplot(gs[1,2])
        ax_f = fig.add_subplot(gs[0,3])
        ax_t = fig.add_subplot(gs[1,3])
    else: 
        ax_p, ax_v, ax_q, ax_w, ax_f, ax_t = in_axs

    ax_p.plot(p[:, 0], p[:, 1], 'g-')
    ax_p.plot(p[:, 0], p[:, 1], 'go')
    if X0 is not None:
        X0.plot(ax_p, color='green', alpha=0.5)
    if Xf is not None:
        Xf.plot(ax_p, color='green', alpha=0.5)
    if ROIs is not None:
        [roi.plot(ax_p, color='blue', alpha=0.5) for roi in ROIs]
    if Obs is not None:
        [obs.plot(ax_p, color='red', alpha=0.5) for obs in Obs]
    ax_p.set_aspect('equal', adjustable='box')
    ax_p.set_xlabel('X (m)')
    ax_p.set_ylabel('Y (m)')
    ax_p.grid()

    ax_v.plot(t, v[:, 0], label='dx')
    ax_v.plot(t, v[:, 1], label='dy')
    ax_v.plot(t, v[:, 2], label='dz')
    ax_v.set_xlabel('Time (s)')
    ax_v.set_ylabel('Velocity (m/s)')
    ax_v.legend()
    ax_v.grid()

    ax_q.plot(t, q[:, 0], 'r', label='q0')
    ax_q.plot(t, q[:, 1], 'g', label='q1')
    ax_q.plot(t, q[:, 2], 'b', label='q2')
    ax_q.plot(t, q[:, 3], 'k', label='q3')
    ax_q.set_xlabel('Time (s)')
    ax_q.set_ylabel('Euler angles (rad)')
    ax_q.legend()
    ax_q.grid()

    ax_w.plot(t, w[:, 0], 'r--', label='d_q0')
    ax_w.plot(t, w[:, 1], 'g--', label='d_q1')
    ax_w.plot(t, w[:, 2], 'b--', label='d_q2')
    ax_w.set_xlabel('Time (s)')
    ax_w.set_ylabel('Angular velocity (rad/s)')
    ax_w.legend()
    ax_w.grid()

    ax_f.plot(t[:u.shape[0]], u[:,0], label='u1')
    ax_f.plot(t[:u.shape[0]], u[:,1], label='u2')
    ax_f.plot(t[:u.shape[0]], u[:,2], label='u3')
    ax_f.axhline(robot.U.lower_bounds[0], color='g', linestyle='-.')#, label="U_lb")
    ax_f.axhline(robot.U.upper_bounds[0], color='g', linestyle='-.')
    if hasattr(robot, 'U_effective'):
        ax_f.axhline(alpha*robot.U_effective.lower_bounds[0], color='b', linestyle='--')#, label="alpha*U_effective_lb")
        ax_f.axhline(alpha*robot.U_effective.upper_bounds[0], color='b', linestyle='--')
        ax_f.axhline(robot.U_effective.lower_bounds[0], color='r', linestyle=':')#, label="U_effective_lb")
        ax_f.axhline(robot.U_effective.upper_bounds[0], color='r', linestyle=':')
    ax_f.set_xlabel('Time (s)')
    ax_f.set_ylabel('Control input (N)')
    ax_f.legend()
    ax_f.grid()

    ax_t.plot(t[:u.shape[0]], u[:,3], label='u4')
    ax_t.plot(t[:u.shape[0]], u[:,4], label='u5')
    ax_t.plot(t[:u.shape[0]], u[:,5], label='u6')
    ax_t.set_xlabel('Time (s)')
    ax_t.set_ylabel('Torque input (Nm)')
    ax_t.legend()
    ax_t.grid()

    if plot:
        plt.savefig(path)
