import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import random

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from Utilities.smarc_modelling.src.smarc_modelling.vehicles.BlueROV import BlueROV
from Utilities.get_reference_trajectory import get_reference_trajectory, ReferenceTrajectory
from Utilities.Robots import FreeFlyer

def plot_instance(fig, axs, uw_robot:BlueROV, sp_robot:FreeFlyer, 
                  x_sol, u_sol, x_ref):
    axs[0].cla()
    axs[0].plot(x_sol[0, :], x_sol[1, :], 'b-', label='Trajectory')
    axs[0].plot(x_ref[0, :], x_ref[1, :], 'r--', label='Reference')
    axs[0].set_xlim([0.0, 8.0])
    axs[0].set_ylim([-1.1, 1.2])
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')
    axs[0].set_title('2D Trajectory')
    axs[0].legend()
    axs[0].grid()

    axs[1].cla()
    times = np.arange(t, t + dt * (N+1), dt)
    axs[1].plot(times, x_sol[7, :], 'b-', label='dx')
    axs[1].plot(times, x_sol[8, :], 'r-', label='dy')
    axs[1].plot(times, x_ref[7, :], 'g--', label='dx_ref')
    axs[1].plot(times, x_ref[8, :], 'm--', label='dy_ref')
    axs[1].set_xlim([0, 45])
    axs[1].set_ylim([-0.2, 0.2])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title('Velocity Profile')
    axs[1].legend()
    axs[1].grid()

    axs[2].cla()
    times = np.arange(t, t + dt * (N), dt)
    axs[2].plot(times, u_sol[0, :], 'b-', label='Control')
    axs[2].plot(times, u_sol[1, :], 'r-', label='Control')
    if uw_robot is not None and sp_robot is not None:
        # we plot the artificial control bounds of the fbl
        lbs, ubs = np.zeros((6,N)), np.zeros((6,N))
        for i in range(N):
            lbs[:, i] = fbl_uw_to_sp(uw_robot, sp_robot, x_sol[:, i], uw_robot.U.lower_bounds)
            ubs[:, i] = fbl_uw_to_sp(uw_robot, sp_robot, x_sol[:, i], uw_robot.U.upper_bounds)
        axs[2].fill_between(times, lbs[0, :], ubs[0, :], color='blue', alpha=0.1)
        axs[2].fill_between(times, lbs[1, :], ubs[1, :], color='red', alpha=0.1)
    axs[2].set_xlim([0, 45])
    axs[2].set_ylim([-80, 80])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Control Input')
    axs[2].set_title('Control Inputs')
    axs[2].grid()

    # show plot and wait
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


def compute_cost(x_vars, u_vars, x_ref, Q, R, P):
    cost = 0
    for i in range(x_vars.shape[1] - 1):
        cost_y = cs.vertcat(
            x_vars[0:3, i] - x_ref[0:3, i],  # Position error
            (x_vars[3:7,i].T@x_ref[3:7,i])**2,
            x_vars[7:10, i] - x_ref[7:10, i],  # Velocity error
            x_vars[10:13, i] - x_ref[10:13, i],  # Angular velocity error
        )
        cost += cost_y.T @ Q @ cost_y + u_vars[:, i].T @ R @ u_vars[:, i]   
    cost_y_e = cs.vertcat(
        x_vars[0:3, -1] - x_ref[0:3, -1],  # Position error
        (x_vars[3:7,-1].T@x_ref[3:7,-1])**2,
        x_vars[7:10, -1] - x_ref[7:10, -1],  # Velocity error
        x_vars[10:13, -1] - x_ref[10:13, -1],  # Angular velocity error
    )
    cost += cost_y_e.T @ P @ cost_y_e
    return cost

# Feedback linearization controller
def fbl_uw_to_sp(uw_robot:BlueROV, sp_robot:FreeFlyer, x,u):
    fx_uw = uw_robot.calculate_fx(x)
    gx_uw = uw_robot.calculate_gx(x)
    fx_sp = sp_robot.calculate_fx(x)
    gx_sp = sp_robot.calculate_gx(x)
    if isinstance(x, cs.SX) | isinstance(x, cs.MX):
        return cs.pinv(gx_sp)@(fx_uw + gx_uw@u - fx_sp)
    else:
        return np.linalg.pinv(gx_sp)@(fx_uw + gx_uw@u - fx_sp)

def fbl_sp_to_uw(uw_robot:BlueROV, sp_robot:FreeFlyer, x,u):
    fx_uw = uw_robot.calculate_fx(x)
    gx_uw = uw_robot.calculate_gx(x)
    fx_sp = sp_robot.calculate_fx(x)
    gx_sp = sp_robot.calculate_gx(x)
    if isinstance(x, cs.SX) | isinstance(x, cs.MX):
        return cs.pinv(gx_uw)@(fx_sp + gx_sp@u - fx_uw)
    else:
        return np.linalg.pinv(gx_uw)@(fx_sp + gx_sp@u - fx_uw)

if __name__ == "__main__":
    # prepare plotting
    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(10, 8))

    # load the reference trajectory
    data = np.load('space_validation/Planning/solutions/bluerov_solution_bezier.npz')
    reference = ReferenceTrajectory(
        r = data['r'],
        dr=data['dr'],
        q=data['q'],
        dt=data['dt']
    )

    x0 = get_reference_trajectory(0.0, reference, order='xyz')
    # x0 = np.array([0., 0., 0., 
    #                1., 0., 0., 0., 
    #                0., 0., 0., 0., 0., 0.])

    dt = 0.25
    N = 15
    real_robot = BlueROV(iX=cs.SX)
    real_sp_robot = FreeFlyer(iX=cs.SX)

    Q = np.diag([               # State weighting matrix
            1e2, 1e2, 1e2,
            1e2,
            3e1, 3e1, 3e1,  
            3e1, 3e1, 3e1])             
    R = 0.1*np.diag([             # Control weighting matrix
            1e0, 1e0, 1e0,
            1e0, 1e0, 1e0]) 
    P = 10 * Q                  # Terminal state weighting matrix
        
    t = 0

    use_fbl = False
    if use_fbl:
        for i in range(100):
            uw_robot = BlueROV(iX=cs.SX)
            sp_robot = FreeFlyer(iX=cs.SX)
            # get reference trajectory
            x_ref = np.zeros((13, N+1))
            ti = t
            for i in range(N+1):
                x_ref[:, i] = get_reference_trajectory(ti, reference, order='xyz').flatten()
                ti += dt

            # create the MPC problem
            ocp = cs.Opti()
            x_vars = ocp.variable(13, N+1)
            u_vars = ocp.variable(6, N)
            delta = ocp.variable(1)

            ocp.set_initial(x_vars, x_ref)
            if 'u_sol' in locals():
                ocp.set_initial(u_vars, u_sol)

            ocp.subject_to(x_vars[:, 0] == x0.flatten())
            ocp.subject_to(delta >= 0)
            for i in range(N):
                ocp.subject_to(u_vars[:, i] <= fbl_uw_to_sp(uw_robot, sp_robot, x_vars[:, i], uw_robot.U.upper_bounds) + delta*np.ones((6,)))
                ocp.subject_to(u_vars[:, i] >= fbl_uw_to_sp(uw_robot, sp_robot, x_vars[:, i], uw_robot.U.lower_bounds) - delta*np.ones((6,)))

            for i in range(N):
                ocp.subject_to(x_vars[:, i+1] == sp_robot.step(x_vars[:, i], u_vars[:, i], dt))

            # set cost
            cost = compute_cost(x_vars, u_vars, x_ref, Q, R, P)
            cost += 1000*delta

            # solve
            ocp.minimize(cost)
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
                'verbose':False, 'ipopt.tol': 1e-4, 'ipopt.max_iter': 1000}
            ocp.solver('ipopt',opts)
            sol = ocp.solve()

            print(f"slack: {sol.value(delta)}")

            x_sol = sol.value(x_vars)
            u_sol = sol.value(u_vars)
            # print(f"x_sol: {x_sol}")
            # print(f"u_sol: {u_sol}")

            plot_instance(fig, axs, real_robot, real_sp_robot, x_sol, u_sol, x_ref)

            # have the robot take a step
            print(f"u_opt (ff): {u_sol[:, 0]}")
            u_applied = fbl_sp_to_uw(real_robot, real_sp_robot, x0, u_sol[:, 0])
            print(f"u_opt (uw): {u_applied}")
            x0 = real_robot.step(x0, u_applied, dt)
            t += dt

    else:
        for i in range(1):
            robot = BlueROV(iX=cs.SX)
            # get reference trajectory
            x_ref = np.zeros((13, N+1))
            ti = t
            for i in range(N+1):
                x_ref[:, i] = get_reference_trajectory(ti, reference, order='xyz').flatten()
                # x_ref[:, i] = np.array([4.54370552e+00,  6.40847734e-02,  1.54554481e+00, 
                #                         1,0,0,0,  
                #                         0,0,0, 
                #                         0,0,0])
                ti += dt

            # create the MPC problem
            ocp = cs.Opti()
            x_vars = ocp.variable(13, N+1)
            u_vars = ocp.variable(6, N)

            ocp.set_initial(x_vars, x_ref)

            ocp.subject_to(x_vars[:, 0] == x0.flatten())
            for i in range(N):
                ocp.subject_to(u_vars[:, i] <= robot.U.upper_bounds)
                ocp.subject_to(u_vars[:, i] >= robot.U.lower_bounds)
            
            for i in range(N):
                ocp.subject_to(x_vars[:, i+1] == robot.step(x_vars[:, i], u_vars[:, i], dt))

            # set cost
            cost = compute_cost(x_vars, u_vars, x_ref, Q, R, P)            

            # solve
            ocp.minimize(cost)
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
                'verbose':False, 'ipopt.tol': 1e-4, 'ipopt.max_iter': 1000}
            ocp.solver('ipopt',opts)
            sol = ocp.solve()

            x_sol = sol.value(x_vars)
            u_sol = sol.value(u_vars)
            # print(f"x_sol: {x_sol}")
            # print(f"u_sol: {u_sol}")

            plot_instance(fig, axs, None, None, x_sol, u_sol, x_ref)

            # have the robot take a step
            x0 = real_robot.step(x0, u_sol[:, 0], dt)
            # add some noise to x0[0:3]
            x0[0:3] += np.random.normal(0, 0.01, size=(3,))
            t += dt

            plt.pause(100)


