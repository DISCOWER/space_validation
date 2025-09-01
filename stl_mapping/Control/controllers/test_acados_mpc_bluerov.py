import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from Utilities.smarc_modelling.src.smarc_modelling.vehicles.BlueROV import BlueROV
from Utilities.Robots import FreeFlyer

from mpc_wrench import MpcWrench
from mpc_fbl_wrench import MpcFBLWrench

def plot_instance(fig, axs, sp_robot:FreeFlyer, uw_robot:BlueROV,
                  sol_sp:dict, sol_uw:dict, x_ref):
    axs[0].cla()
    axs[0].plot(x_ref[0, :], x_ref[1, :], 'ro', label='Reference')
    axs[0].plot(sol_sp['x'][0, :], sol_sp['x'][1, :], 'b-', label='Space Traj.')
    axs[0].plot(sol_uw['x'][0, :], sol_uw['x'][1, :], 'r--', label='Underwater Traj.')
    axs[0].set_xlim([0.0, 8.0])
    axs[0].set_ylim([-2, 2])
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')
    axs[0].set_title('2D Trajectory')
    axs[0].legend()
    axs[0].grid()

    axs[1].cla()
    times = np.arange(t, t + dt * (N+1), dt)
    axs[1].plot(times[0:x_ref.shape[1]], x_ref[7, :], 'k--', label='dx_ref')
    axs[1].plot(times[0:x_ref.shape[1]], x_ref[8, :], 'k--', label='dy_ref')
    axs[1].plot(times[0:sol_sp['x'].shape[1]], sol_sp['x'][7, :], 'b-', label='Space dx')
    axs[1].plot(times[0:sol_sp['x'].shape[1]], sol_sp['x'][8, :], 'r-', label='Space dy')
    axs[1].plot(times[0:sol_uw['x'].shape[1]], sol_uw['x'][7, :], 'r--', label='Underwater dx')
    axs[1].plot(times[0:sol_uw['x'].shape[1]], sol_uw['x'][8, :], 'b--', label='Underwater dy')
    axs[1].set_xlim([0, 10])
    axs[1].set_ylim([-1.2, 1.2])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title('Velocity Profile')
    axs[1].legend()
    axs[1].grid()

    axs[2].cla()
    times = np.arange(t, t + dt * (N+1), dt)
    axs[2].axhline(x_ref[3,0], color='k', linestyle='--')
    axs[2].axhline(x_ref[4,0], color='k', linestyle='--')
    axs[2].axhline(x_ref[5,0], color='k', linestyle='--')
    axs[2].axhline(x_ref[6,0], color='k', linestyle='--')
    axs[2].plot(times[0:sol_sp['x'].shape[1]], sol_sp['x'][3, :], 'b-', label='Space qw')
    axs[2].plot(times[0:sol_sp['x'].shape[1]], sol_sp['x'][4, :], 'r-', label='Space qx')
    axs[2].plot(times[0:sol_sp['x'].shape[1]], sol_sp['x'][5, :], 'g-', label='Space qy')
    axs[2].plot(times[0:sol_sp['x'].shape[1]], sol_sp['x'][6, :], 'm-', label='Space qz')
    axs[2].plot(times[0:sol_uw['x'].shape[1]], sol_uw['x'][3, :], 'b--', label='Underwater qw')
    axs[2].plot(times[0:sol_uw['x'].shape[1]], sol_uw['x'][4, :], 'r--', label='Underwater qx')
    axs[2].plot(times[0:sol_uw['x'].shape[1]], sol_uw['x'][5, :], 'g--', label='Underwater qy')
    axs[2].plot(times[0:sol_uw['x'].shape[1]], sol_uw['x'][6, :], 'm--', label='Underwater qz')
    axs[2].set_xlim([0, 10])
    axs[2].axhline(1, color='k', linestyle='--', linewidth=0.8)
    axs[2].axhline(-1, color='k', linestyle='--', linewidth=0.8)
    # axs[2].set_ylim([-1.2, 1.2])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Angular Position (rad)')
    axs[2].set_title('Angular Position Profile')
    axs[2].legend()
    axs[2].grid()

    axs[3].cla()
    times = np.arange(t, t + dt * (N), dt)
    axs[3].plot(times[0:sol_sp['u'].shape[1]], sol_sp['u'][0, :], 'b-', label='Control')
    axs[3].plot(times[0:sol_sp['u'].shape[1]], sol_sp['u'][1, :], 'r-', label='Control')
    axs[3].plot(times[0:sol_uw['u'].shape[1]], sol_uw['u'][0, :], 'b--', label='Control')
    axs[3].plot(times[0:sol_uw['u'].shape[1]], sol_uw['u'][1, :], 'r--', label='Control')
    axs[3].axhline(sp_robot.U.lower_bounds[0], color='k', linestyle='-')
    axs[3].axhline(sp_robot.U.upper_bounds[0], color='k', linestyle='-')
    axs[3].axhline(uw_robot.U.lower_bounds[0], color='k', linestyle='--')
    axs[3].axhline(uw_robot.U.upper_bounds[0], color='k', linestyle='--')
    axs[3].set_xlim([0, 10])
    axs[3].set_ylim([-100, 100])
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Control Input')
    axs[3].set_title('Control Inputs')
    axs[3].grid()

    # show plot and wait
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

if __name__ == "__main__":
    # prepare plotting
    plt.ion()
    fig, axs = plt.subplots(1, 4, figsize=(15, 6))

    # robot_name = 'atmos'
    robot_name = 'bluerov'

    # mpc = MpcWrench(model_name=robot_name)
    mpc = MpcFBLWrench(model_name=robot_name)

    dt = mpc.dt
    t = 0.
    N = mpc.Nx

    # x0 = np.array([0., 0., 0., 
    #                1/np.sqrt(2), 0., 0., 1/np.sqrt(2), 
    #                0., 0., 0., 
    #                0., 0., 0.])
    x0 = np.array([3, 0, 1.5, 
                #    1, 0, 0, 0,
                   1/np.sqrt(2), 0., 0., 1/np.sqrt(2), 
                #    -0.74473321,  0.00509102,  0.01644424, -0.66714031,  
                #    0.01992397,  0.13478149, -0.0642853,   
                #    0.04263152,  0.18871628, -0.3909702])
                     0., 0., 0.,
                     0., 0., 0.])

    x_ref = np.array([3.55000000e+00,  5.50000008e-01,  1.55000000e+00,  
                      1, 0,0,0,
                    #   7.07106781e-01, -1.33906769e-22, -9.69625088e-23,  7.07106781e-01,
                     0, 0, 0,
                     0, 0, 0,
                     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
    # x_ref = np.array([3.55000000e+00,  5.50000008e-01,  1.55000000e+00,  1,
    #                 0, 0, 0,
    #                  3.43406489e-02, -3.85964402e-02, -5.51617376e-03,
    #                  -5.23438675e-03, -3.83480741e-02, -4.37074084e-02,
    #                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
    x_ref = np.repeat(x_ref, mpc.Nx).reshape(13+6, mpc.Nx)

    uw_robot = BlueROV(iX=cs.SX)
    sp_robot = FreeFlyer(iX=cs.SX)

    for i in range(100):
        u_sol, x_sol, sol_sp, sol_uw = mpc.get_input(x0, x_ref)

        plot_instance(fig, axs, sp_robot, uw_robot, 
                      sol_sp, sol_uw, x_ref)

        x0 = uw_robot.step(x0, 1.2*u_sol, dt)
        t += dt
        
        plt.pause(0.05)


