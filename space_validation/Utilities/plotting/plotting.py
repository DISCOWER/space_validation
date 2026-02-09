import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Utilities.rotations import euler_to_quat_np, quat_to_euler_np
from scipy.spatial.transform import Rotation as R

def plot_planning_results(robot, t, x, u, 
                          X0=None, Xf=None, ROIs=None, Obs=None, alpha=1.0,
                          in_axs = None, plot=True, D3=False,
                          path="space_validation/Planning/figures/sp_trajectory.png"):
    # if x.shape[1] == 12:
    #     p, e, v, w = x[:, 0:3], x[:, 3:6], x[:, 6:9], x[:, 9:12]
    #     q = np.zeros((x.shape[0], 4))
    #     for i in range(x.shape[0]):
    #         q[i, :] = euler_to_quat_np(e[i, :], order='zyx')
    #     x = np.hstack((p, q, v, w))
    #     print(f"Converted state representation:\n{x}")
    # elif x.shape[1] == 13:
    #     p, q, v, w = x[:, 0:3], x[:, 3:7], x[:, 7:10], x[:, 10:13]
    # else:
    #     raise ValueError("State vector x must have 12 or 13 columns.")
    if x.shape[1] == 12:
        p_idxs = [0, 1, 2]
        q_idxs = [3, 4, 5]
        v_idxs = [6, 7, 8]
        w_idxs = [9, 10, 11]
    elif x.shape[1] == 13:
        p_idxs = [0, 1, 2]
        q_idxs = [3, 4, 5, 6]
        v_idxs = [7, 8, 9]
        w_idxs = [10, 11, 12]
    else:
        raise ValueError("State vector x must have 12 or 13 columns.")
    
    p = x[:, p_idxs]
    q = x[:, q_idxs]
    v = x[:, v_idxs]
    w = x[:, w_idxs]

    if q.shape[1] == 4:  # If quaternion representation
        heading = quat_to_euler_np(q, order='xyz')[:, 2]
    else:  # If Euler angles
        heading = q[:, 2]  # Assuming z-axis is up

    # plot the trajectory
    if in_axs is None:
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(2,5, figure=fig)
        if D3:
            ax_p = fig.add_subplot(gs[:,0:2], projection='3d')
            ax_p.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1
        else:
            ax_p = fig.add_subplot(gs[:,0:2])
        ax_p2 = fig.add_subplot(gs[0,2])
        ax_v = fig.add_subplot(gs[1,2])
        ax_q = fig.add_subplot(gs[0,3])
        ax_w = fig.add_subplot(gs[1,3])
        ax_f = fig.add_subplot(gs[0,4])
        ax_t = fig.add_subplot(gs[1,4])
    else: 
        ax_p, ax_v, ax_q, ax_w, ax_f, ax_t = in_axs

    if D3:
        ax_p.plot(p[:, 0], p[:, 1], p[:, 2], 'g-')
        # create arrows from pitch, roll, yaw angles
        forward = np.array([0.5, 0, 0])
        for i in range(len(p)):
            if q.shape[1] == 4:
                Rot = R.from_quat(q[i, :],scalar_first=True)
            else:
                Rot = R.from_euler('yxz', q[i, :], degrees=False)
            direction = Rot.apply(forward)
            # print(f"direction: {direction}")
            ax_p.quiver(p[i, 0], p[i, 1], p[i, 2], 
                        direction[0], direction[1], direction[2], color='k')

        if X0 is not None:
            X0.plot(ax_p, color='green', alpha=0.5)
        if Xf is not None:
            Xf.plot(ax_p, color='green', alpha=0.5)
        if Obs is not None:
            [obs.plot(ax_p, color='red', alpha=0.5) for obs in Obs]
        if ROIs is not None:
            [roi.plot(ax_p, color='blue', alpha=0.5) for roi in ROIs]

        limits = np.array([
            ax_p.get_xlim3d(),
            ax_p.get_ylim3d(),
            ax_p.get_zlim3d()
        ])
        spans = limits[:,1] - limits[:,0]
        centers = np.mean(limits, axis=1)
        radius = 0.5 * max(spans)
        ax_p.set_xlim3d([centers[0] - radius, centers[0] + radius])
        ax_p.set_ylim3d([centers[1] - radius, centers[1] + radius])
        ax_p.set_zlim3d([centers[2] - radius, centers[2] + radius])
    else:
        ax_p.plot(p[:, 0], p[:, 1], 'g-')
        # ax_p.plot(p[:, 0], p[:, 1], 'go')
        if heading is not None:
            ax_p.quiver(p[:, 0], p[:, 1], np.cos(heading), np.sin(heading), color='r', scale=10)
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

    ax_p2.plot(t, p[:, 0], 'r-',label='dx')
    ax_p2.plot(t, p[:, 1], 'g-', label='dy')
    ax_p2.plot(t, p[:, 2], 'b-', label='dz')
    # ax_p2.set_xlabel('Time (s)')
    ax_p2.set_ylabel('Position (m)')
    ax_p2.legend()
    ax_p2.grid()

    ax_v.plot(t, v[:, 0], 'r-',label='dx')
    # ax_v.plot(t, v[:, 0], 'ro')
    ax_v.plot(t, v[:, 1], 'g-', label='dy')
    # ax_v.plot(t, v[:, 1], 'go')
    ax_v.plot(t, v[:, 2], 'b-', label='dz')
    # ax_v.plot(t, v[:, 2], 'bo')
    ax_v.set_xlabel('Time (s)')
    ax_v.set_ylabel('Velocity (m/s)')
    ax_v.legend()
    ax_v.grid()

    ax_q.plot(t, q[:, 0], 'r', label='q0')
    ax_q.plot(t, q[:, 1], 'g', label='q1')
    ax_q.plot(t, q[:, 2], 'b', label='q2')
    if q.shape[1] == 4:  # If quaternion representation
        ax_q.plot(t, q[:, 3], 'k', label='q3')
    # ax_q.set_xlabel('Time (s)')
    if q.shape[1] == 4:
        ax_q.set_ylabel('Quaternion (q0, q1, q2, q3)')
    else:
        ax_q.set_ylabel('Euler angles (pitch roll yaw)')
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
    # ax_f.axhline(robot.U.lower_bounds[0], color='g', linestyle='-.')#, label="U_lb")
    # ax_f.axhline(robot.U.upper_bounds[0], color='g', linestyle='-.')
    # ax_f.axhline(robot.calculate_U_effective(np.zeros((13,)),alpha).lower_bounds[0], color='b', linestyle='--')
    # ax_f.axhline(robot.calculate_U_effective(np.zeros((13,)),alpha).upper_bounds[0], color='b', linestyle='--')
    # ax_f.set_xlabel('Time (s)')
    ax_f.set_ylabel('Control input (N)')
    ax_f.legend()
    ax_f.grid()

    ax_t.plot(t[:u.shape[0]], u[:,3], label='u4')
    ax_t.plot(t[:u.shape[0]], u[:,4], label='u5')
    ax_t.plot(t[:u.shape[0]], u[:,5], label='u6')
    # ax_t.axhline(robot.U.lower_bounds[3], color='g', linestyle='-.')#, label="U_lb")
    # ax_t.axhline(robot.U.upper_bounds[3], color='g', linestyle='-.')
    # ax_t.axhline(robot.calculate_U_effective(np.zeros((13,)),alpha).lower_bounds[3], color='b', linestyle='--')
    # ax_t.axhline(robot.calculate_U_effective(np.zeros((13,)),alpha).upper_bounds[3], color='b', linestyle='--')
    ax_t.set_xlabel('Time (s)')
    ax_t.set_ylabel('Torque input (Nm)')
    ax_t.legend()
    ax_t.grid()

    if plot:
        plt.savefig(path)
    else:
        plt.show()


def animate_pose_trace(x, interval=50, axis_len=0.2):
    """
    Animate a trajectory with position and quaternion orientation.


    Args:
    x: numpy array of shape (N, 7)
    where each row is [px, py, pz, qx, qy, qz, qw]
    interval: delay between frames in ms (default 50)
    axis_len: length of orientation axes to draw (default 0.2)
    """
    assert x.shape[1] >= 7, "Input must be of shape (N, 7): [pos(3), quat(4)]"

    pos = x[:, :3]
    quats = x[:, 3:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory line
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'k--', alpha=0.3)

    # Set axis limits
    max_range = np.ptp(pos, axis=0).max() / 2
    mid = pos.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Initialize plot elements
    point, = ax.plot([], [], [], 'bo')
    x_axis, = ax.plot([], [], [], 'r-', lw=2)
    y_axis, = ax.plot([], [], [], 'g-', lw=2)
    z_axis, = ax.plot([], [], [], 'b-', lw=2)

    def init():
        point.set_data([], [])
        point.set_3d_properties([])
        for line in (x_axis, y_axis, z_axis):
            line.set_data([], [])
            line.set_3d_properties([])
        return point, x_axis, y_axis, z_axis

    def update(frame):
        p = pos[frame]
        q = quats[frame]


        # Rotation matrix from quaternion
        Rmat = R.from_quat(q).as_matrix()
        axes = Rmat * axis_len


        origin = p.reshape(3, 1)


        point.set_data(p[0:2])
        point.set_3d_properties(p[2])


        x_axis.set_data([p[0], p[0] + axes[0, 0]], [p[1], p[1] + axes[1, 0]])
        x_axis.set_3d_properties([p[2], p[2] + axes[2, 0]])


        y_axis.set_data([p[0], p[0] + axes[0, 1]], [p[1], p[1] + axes[1, 1]])
        y_axis.set_3d_properties([p[2], p[2] + axes[2, 1]])


        z_axis.set_data([p[0], p[0] + axes[0, 2]], [p[1], p[1] + axes[1, 2]])
        z_axis.set_3d_properties([p[2], p[2] + axes[2, 2]])


        return point, x_axis, y_axis, z_axis


    ani = FuncAnimation(fig, update, frames=len(pos), init_func=init,
    blit=True, interval=interval)
    plt.show()
    return ani