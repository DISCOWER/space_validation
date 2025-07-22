import numpy as np
import matplotlib.pyplot as plt

def plot_planning_results(robot, t, x, u, X0, Xf, ROIs, Obs, alpha=1.0,
                          path="stl_mapping/Planning/figures/sp_trajectory.png"):
    # plot the trajectory
    fig, axs = plt.subplots(1,4, figsize=(15, 5))
    axs[0].plot(x[:, 0], x[:, 1], 'g-')
    axs[0].plot(x[:, 0], x[:, 1], 'go')
    X0.plot(axs[0], color='green', alpha=0.5)
    Xf.plot(axs[0], color='green', alpha=0.5)
    [roi.plot(axs[0], color='blue', alpha=0.5) for roi in ROIs]
    [obs.plot(axs[0], color='red', alpha=0.5) for obs in Obs]
    axs[0].set_aspect('equal', adjustable='box')

    axs[1].plot(t, x[:, 6], label='dx')
    axs[1].plot(t, x[:, 7], label='dy')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t, x[:, 3], 'r', label='yaw')
    axs[2].plot(t, x[:, 4], 'g', label='roll')
    axs[2].plot(t, x[:, 5], 'b', label='pitch')
    axs[2].plot(t, x[:, 9], 'r--', label='d_yaw')
    axs[2].plot(t, x[:, 10], 'g--', label='d_roll')
    axs[2].plot(t, x[:, 11], 'b--', label='d_pitch')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Euler angles (rad)')
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(t, u[:,0])
    axs[3].plot(t, u[:,1])
    axs[3].plot(t, u[:,2])
    axs[3].axhline(robot.U.lower_bounds[0], color='g', linestyle='-.', label="U_lb")
    axs[3].axhline(robot.U.upper_bounds[0], color='g', linestyle='-.')
    if hasattr(robot, 'U_effective'):
        axs[3].axhline(alpha*robot.U_effective.lower_bounds[0], color='b', linestyle='--', label="alpha*U_effective_lb")
        axs[3].axhline(alpha*robot.U_effective.upper_bounds[0], color='b', linestyle='--')
        axs[3].axhline(robot.U_effective.lower_bounds[0], color='r', linestyle=':', label="U_effective_lb")
        axs[3].axhline(robot.U_effective.upper_bounds[0], color='r', linestyle=':')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Control input (N)')
    plt.savefig(path)
