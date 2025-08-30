#!/usr/bin/env python3
import os
from pathlib import Path
from rosbags.rosbag2 import Reader
import csv
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial.transform import Rotation as R
from matplotlib.transforms import Affine2D
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utilities.sets import HyperRectangle

class RosBagClass():
    def __init__(self,
                 file_path:str,
                 robot_name:str='atmos',
                 threeD:bool=False):
        self.file_path = file_path
        self.robot_name = robot_name
        self.threeD = threeD

        if self.robot_name == 'atmos':
            self.robot_ns = 'snap'
        elif self.robot_name == 'bluerov':
            self.robot_ns = 'itrl_rov_1'

        self.robot_img = plt.imread(f'{str(Path.home())}/space_ws/src/stl_mapping/stl_mapping/Utilities/plotting/assets/{self.robot_name}.png')        

        self.topics = ["_".join([self.robot_ns,'fmu','out','vehicle_angular_velocity']),
                       "_".join([self.robot_ns,'fmu','out','vehicle_attitude']),
                       "_".join([self.robot_ns,'fmu','out','vehicle_local_position']),
                       "_".join([self.robot_ns,'fmu','in','vehicle_thrust_setpoint']),
                       "_".join([self.robot_ns,'fmu','in','vehicle_torque_setpoint']),
                       "_".join([self.robot_ns,'disturbance_estimate'])]

        self.topics_data = {topic: pd.read_csv(os.path.join(self.file_path, f'{topic}.csv')) for topic in self.topics}

        self.plan = {}
        self.read_plans()

    def read_plans(self):
        self.plan = np.load(os.path.join(self.file_path, f'solutions/atmos_nonlinear_solution.npz'))

    def get_topic_data(self,topic_name):
        return next((v for k,v in self.topics_data.items() if topic_name in k), None)
    
    #! Plotting individual topics
    def plot_position(self,ax,plot_robot:int=0):
        vehicle_local_position = self.get_topic_data('vehicle_local_position')

        ax.plot(self.plan['x'][:,0], self.plan['x'][:,1], 'r--', label=r'$p_{ref}$')
        ax.plot(vehicle_local_position['y'], vehicle_local_position['x'], label=r'$p$')
        if plot_robot > 1:
            # plot the robot image at the position, 'plot_robot' times interpolated
            # along the executed trajectory
            alphas = np.linspace(0, 1, plot_robot+1)[1::]
            x, y = vehicle_local_position['y'], vehicle_local_position['x']
            vehicle_attitude = self.get_topic_data('vehicle_attitude')
            q = vehicle_attitude['q']
            indices = np.round(np.linspace(0, len(x)-1, plot_robot)).astype(int)
            for cnt, i in enumerate(indices):
                x_i, y_i = x[i], y[i]
                Rot = R.from_quat(np.fromstring(q[i].strip("[]"), sep=" "),scalar_first=True)    
                yaw = Rot.as_euler('xyz', degrees=True)[2]
                im = ax.imshow(self.robot_img, extent=[x_i - 0.25, x_i + 0.25, y_i - 0.25, y_i + 0.25], origin='upper',alpha=alphas[cnt])
                # now transform accordingly
                trans_data = (
                    Affine2D()
                    .rotate_deg_around(x_i, y_i, yaw)      # shift so that image center is at (x,y)
                    + ax.transData
                )
                im.set_transform(trans_data)

        ax.set_title('Position')
        ax.set_xlabel('x position (m)')
        ax.set_ylabel('y position (m)')
        ax.axis('equal')
        ax.grid()
        ax.legend()

    def plot_position_time(self,ax):
        vehicle_local_position = self.get_topic_data('vehicle_local_position')
        rosbag_time = (vehicle_local_position['timestamp']-vehicle_local_position['timestamp'][0])/1e6
        ax.plot(self.plan['times'], self.plan['x'][:,0], 'r--')#, label=r'$x_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,1], 'g--')#, label=r'$y_{ref}$')
        if self.threeD:
            ax.plot(self.plan['times'], self.plan['x'][:,2], 'b--')#, label=r'$z_{ref}$')
        ax.plot(rosbag_time, vehicle_local_position['y'], 'r-', label=r'$x$')
        ax.plot(rosbag_time, vehicle_local_position['x'], 'g-', label=r'$y$')
        if self.threeD:
            ax.plot(rosbag_time, vehicle_local_position['z'], 'b-', label=r'$z$')

        # find the maximal spatial deviation
        x_ex = np.array([vehicle_local_position['y'], vehicle_local_position['x'], vehicle_local_position['z']])
        x_ref_interp = np.array([np.interp(np.array(rosbag_time), self.plan['times'], self.plan['x'][:,i]) for i in range(3)])
        D = np.abs(x_ex - x_ref_interp)
        max_dev, max_idx, max_dim = -np.inf, 0, 0
        for i, d in enumerate(D.T):
            if np.max(d) > max_dev:
                max_dev = np.max(d)
                max_idx = i
                max_dim = np.argmax(d)
        t = rosbag_time[max_idx]
        print(f"t: {t}")

        ax.plot([t,t], [min(x_ref_interp[max_dim,max_idx], x_ex[max_dim,max_idx]), 
                        max(x_ref_interp[max_dim,max_idx], x_ex[max_dim,max_idx])],'k--',linewidth=3)
        # and add text with the value of it $\rho_{\phi}=max_dev$
        ax.text(t, max(x_ref_interp[max_dim,max_idx], x_ex[max_dim,max_idx]), f"$\\rho_{{\\phi}}={max_dev:.2f}$", fontsize=12, ha='center')

        ax.set_title('Position over Time')
        # ax.set_xlabel('Time (s)')
        ax.set_ylabel('position (m)')
        ax.grid()
        ax.legend()

    def plot_velocity(self,ax):
        vehicle_local_position = self.get_topic_data('vehicle_local_position')
        rosbag_time = (vehicle_local_position['timestamp']-vehicle_local_position['timestamp'][0])/1e6

        ax.plot(self.plan['times'], self.plan['x'][:,7], 'r--', label=r'$\dot{x}_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,8], 'g--', label=r'$\dot{y}_{ref}$')
        if self.threeD:
            ax.plot(self.plan['times'], self.plan['x'][:,9], 'b--', label=r'$\dot{z}_{ref}$')
        ax.plot(rosbag_time, vehicle_local_position['vx'], 'r-', label=r'$\dot{x}$')
        ax.plot(rosbag_time, vehicle_local_position['vy'], 'g-', label=r'$\dot{y}$')
        if self.threeD:
            ax.plot(rosbag_time, vehicle_local_position['vz'], 'b-', label=r'$\dot{z}$')
        ax.set_title('Linear Velocity')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('velocity (m/s)')
        ax.set_ylim([-0.2, 0.5])
        ax.grid()
        # ax.legend()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

    def plot_attitude(self,ax):
        vehicle_attitude = self.get_topic_data('vehicle_attitude')
        rosbag_time = (vehicle_attitude['timestamp']-vehicle_attitude['timestamp'][0])/1e6

        q = np.array([np.fromstring(q_i.strip("[]"), sep=" ") for q_i in vehicle_attitude['q']])

        ax.plot(self.plan['times'], self.plan['x'][:,3], 'r--')#, label=r'$q^w_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,4], 'g--')#, label=r'$q^x_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,5], 'b--')#, label=r'$q^y_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,6], 'c--')#, label=r'$q^z_{ref}$')
        ax.plot(rosbag_time, q[:,0], 'r-', label=r'$q^w$')
        ax.plot(rosbag_time, q[:,1], 'g-', label=r'$q^x$')
        ax.plot(rosbag_time, q[:,2], 'b-', label=r'$q^y$')
        ax.plot(rosbag_time, q[:,3], 'c-', label=r'$q^z$')
        ax.set_title('Attitude')
        # ax.set_xlabel('time (s)')
        ax.set_ylabel('quaternion')
        ax.set_ylim([-1, 2.0])
        ax.grid()
        # horizontal legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

    def plot_angular_velocity(self,ax):
        vehicle_angular_velocity = self.get_topic_data('vehicle_angular_velocity')
        rosbag_time = (vehicle_angular_velocity['timestamp']-vehicle_angular_velocity['timestamp'][0])/1e6

        w = np.array([np.fromstring(w_i.strip("[]"), sep=" ") for w_i in vehicle_angular_velocity['xyz']])
        if self.threeD:
            ax.plot(self.plan['times'], self.plan['x'][:,10], 'g--', label=r'$\theta_{ref}$')
            ax.plot(self.plan['times'], self.plan['x'][:,11], 'r--', label=r'$\phi_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,12], 'b--', label=r'$\psi_{ref}$')
        if self.threeD:
            ax.plot(rosbag_time, w[:,0], 'r-', label=r'$\phi$')
            ax.plot(rosbag_time, w[:,1], 'g-', label=r'$\theta$')
        ax.plot(rosbag_time, w[:,2], 'b-', label=r'$\psi$')
        ax.set_title('Angular Velocity')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('angular velocity (rad/s)')
        ax.set_ylim([-0.2,0.3])
        ax.grid()
        # ax.legend()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

    def plot_force(self,ax):
        vehicle_thrust_setpoint = self.get_topic_data('vehicle_thrust_setpoint')
        rosbag_time = (vehicle_thrust_setpoint['timestamp']-vehicle_thrust_setpoint['timestamp'][0])/1e6

        f = np.array([np.fromstring(w_i.strip("[]"), sep=" ") for w_i in vehicle_thrust_setpoint['xyz']])
        ax.plot(rosbag_time, f[:,0], 'b-', label=r'$f_x$')
        ax.plot(rosbag_time, f[:,1], 'g-', label=r'$f_y$')
        if self.threeD:
            ax.plot(rosbag_time, f[:,2], 'r-', label=r'$f_z$')
        ax.set_title('Thrust Setpoint')
        # ax.set_xlabel('time (s)')
        ax.set_ylabel('thrust (N)')
        ax.grid()
        ax.legend()

    def plot_disturbance(self,ax,sigma=1):
        disturbance_estimate = self.get_topic_data('disturbance_estimate')
        # TODO: ensure time is properly recorded in message
        # rosbag_time = np.array([float(t) for t in disturbance_estimate['time']])
        # rosbag_time = (rosbag_time - rosbag_time[0])

        # dx = disturbance_estimate['twist.twist.linear.x']
        # dy = disturbance_estimate['twist.twist.linear.y']
        # dz = disturbance_estimate['twist.twist.linear.z']
        # cov = disturbance_estimate['twist.covariance']

        # TODO: temporary fake data to show what we want to show
        def brownian_motion(n_steps=1000, T=1.0, mu=0.0, sigma=0.1):
            dt = T / n_steps
            # Gaussian increments
            increments = np.random.normal(loc=mu*dt, scale=np.sqrt(dt)*sigma, size=n_steps)
            # Cumulative sum to get Brownian path
            W = np.cumsum(increments)
            # Start at 0
            W = np.insert(W, 0, 0.0)
            return W
        dx = brownian_motion(n_steps=100, T=1.0, mu=0.0)
        dy = brownian_motion(n_steps=100, T=1.0, mu=0.0)
        dz = brownian_motion(n_steps=100, T=1.0, mu=0.0)
        cov_x = 0.1
        cov_y = 0.15
        cov_z = 0.1

        ax.plot(dx, 'b-', label=r'$d_x$')
        ax.fill_between(np.arange(len(dx)), dx - sigma*cov_x, dx + sigma*cov_x, color='b', alpha=0.2)
        ax.plot(dy, 'g-', label=r'$d_y$')
        ax.fill_between(np.arange(len(dy)), dy - sigma*cov_y, dy + sigma*cov_y, color='g', alpha=0.2)
        if self.threeD:
            ax.plot(dz, 'r-', label=r'$d_z$')
            ax.fill_between(np.arange(len(dz)), dz - sigma*cov_z, dz + sigma*cov_z, color='r', alpha=0.2)
        # vertical lines of \alpha D
        ax.axhline(-0.8, color='k', linestyle='--')
        ax.axhline(0.8, color='k', linestyle='--')
        ax.set_ylim([-1,1])
        ax.set_title('Estimated Disturbance')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('disturbance (N)')
        ax.grid()
        ax.legend()
