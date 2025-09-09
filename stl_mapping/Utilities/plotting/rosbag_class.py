#!/usr/bin/env python3
import os, sys
from pathlib import Path
from rosbags.rosbag2 import Reader
import csv
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial.transform import Rotation as R
from matplotlib.transforms import Affine2D
import ast
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from Utilities.sets import HyperRectangle
from Utilities.Robots import FreeFlyer
from Utilities.smarc_modelling.src.smarc_modelling.vehicles.BlueROV import BlueROV

class RosBagClass():
    def __init__(self,
                 csv_file:pd.DataFrame,
                 plan:dict,
                 robot_name:str='atmos',
                 threeD:bool=False,
                 offset:np.ndarray=np.array([0,0,0]),
                 exp:int=1):
        self.csv_file = csv_file
        self.robot_name = robot_name
        self.threeD = threeD
        self.plan = plan
        self.offset = offset

        print(f"alpha: {self.plan['alpha']}")
        print(f"dt: {self.plan['dt']}")

        if self.robot_name == 'atmos':
            self.robot_ns = 'snap'
            self.robot = FreeFlyer()
        elif self.robot_name == 'bluerov':
            self.robot_ns = 'itrl_rov_1'
            self.robot = BlueROV()

        if exp == 1:
            X0 = HyperRectangle(center=np.array([0,1.5, 0]),                   size=np.array([0.5, 0.5, 0.5]), color='g', name=r'X0')
            Xf = HyperRectangle(center=np.array([0,4.0, 0]),                   size=np.array([0.5, 0.5, 0.5]), color='g', name=r'Xf')
            XA = HyperRectangle(center=np.array([-0.5,2.25, 0,  -np.pi/2]),    size=np.array([0.5, 0.5, 0.5,  np.pi/4]),  color='r', name=r'A')
            XB = HyperRectangle(center=np.array([0.5, 2.0, 0,  np.pi/2]),       size=np.array([0.5, 0.5, 0.5,  np.pi/4]),  color='b', name=r'B')
            XC1 = HyperRectangle(center=np.array([0.5, 3.0, 0,  np.pi/2]),      size=np.array([0.5, 0.5, 0.5,  np.pi/4]),  color='w', name=r'C_1')
            XC2 = HyperRectangle(center=np.array([-0.5, 3.25, 0,  -np.pi/2]),    size=np.array([0.5, 0.5, 0.5,  np.pi/4]), color='w', name=r'C_2')
            self.Xs = [X0, Xf, XA, XB, XC1, XC2]
        elif exp == 2:
            depth = 0
            X0 = HyperRectangle(center=np.array([0, 2.0, depth]), size=np.array([0.5, 0.5, 0.5]), color='g', name=r'X_0')
            Obs1 = HyperRectangle(center=np.array([0.0, 4.0, depth]), size=np.array([0.5, 0.5, 0.5]), color='r', name=r'Obs')
            Xfront  = HyperRectangle(center=np.array([0, 3.0, depth, 0, 0]), size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
            Xback1   = HyperRectangle(center=np.array([0, 5.0, depth, 0, np.pi]), size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
            Xleft1   = HyperRectangle(center=np.array([-0.75, 4.0, depth, 0, np.pi/2]), size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
            Xright1  = HyperRectangle(center=np.array([0.75, 4.0, depth, 0, -np.pi/2]), size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
            Xtop1    = HyperRectangle(center=np.array([0, 4.0, depth+0.75, np.pi/2, 0]), size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
            Xtop2    = HyperRectangle(center=np.array([0, 4.0, depth+0.75, -3*np.pi/2, 0]), size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
            Xbottom1 = HyperRectangle(center=np.array([0, 4.0, depth-0.75, np.deg2rad(-80), 0]), size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
            Xbottom2 = HyperRectangle(center=np.array([0, 4.0, depth-0.75, np.deg2rad(260), 0]), size=np.array([0.5, 0.5, 0.5, np.pi/8, np.pi/8]))
            self.Xs = [X0,Xfront, Xback1, Xleft1, Xright1, Obs1]
        else:
            raise ValueError("Experiment not recognized.")
            
        self.robot_img = plt.imread(f'{str(Path.home())}/space_ws/src/stl_mapping/stl_mapping/Utilities/plotting/assets/{self.robot_name}.png')   
        self.get_topic_data()     

    def get_topic_data(self):
        topics = {}

        def get_data(topic, fields):
            data = {field: [] for field in fields}
            for field in fields:
                if topic + field in self.csv_file.columns:
                    data[field] = self.csv_file[topic + field].dropna().to_numpy()
                else:
                    print(f"Warning: Field '{topic + field}' not found in CSV file.")
                    data[field] = np.array([])
            return data

        topic = f'/{self.robot_ns}/fmu/out/vehicle_local_position_v1/'
        fields = ['timestamp','x','y','z','vx','vy','vz']
        topics['vehicle_local_position'] = get_data(topic, fields)
        # fix xyz with offset
        topics['vehicle_local_position']['x'] -= self.offset[0]
        topics['vehicle_local_position']['y'] -= self.offset[1]
        topics['vehicle_local_position']['z'] -= self.offset[2]

        topic = f'/{self.robot_ns}/fmu/out/vehicle_attitude/'
        fields = ['timestamp','q[0]','q[1]','q[2]','q[3]']
        topics['vehicle_attitude'] = get_data(topic, fields)

        topic = f'/{self.robot_ns}/fmu/out/vehicle_angular_velocity/'
        fields = ['timestamp','xyz[0]','xyz[1]','xyz[2]']
        topics['vehicle_angular_velocity'] = get_data(topic, fields)

        # Vehicle thrust and torque setpoints
        topic = f'/{self.robot_ns}/fmu/in/vehicle_thrust_setpoint/'
        fields = ['timestamp','xyz[0]','xyz[1]','xyz[2]']
        topics['vehicle_thrust_setpoint'] = get_data(topic, fields)

        topic = f'/{self.robot_ns}/fmu/in/vehicle_torque_setpoint/'
        fields = ['timestamp','xyz[0]','xyz[1]','xyz[2]']
        topics['vehicle_torque_setpoint'] = get_data(topic, fields)

        # Pre- and post feedback equivalence control setpoints
        topic = f'/stl_mapping/pre_fbl_force_setpoint/'
        fields = ['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]']
        topics['pre_fbl_force_setpoint'] = get_data(topic, fields)

        topic = f'/stl_mapping/pre_fbl_torque_setpoint/'
        fields = ['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]']
        topics['pre_fbl_torque_setpoint'] = get_data(topic, fields)

        topic = f'/stl_mapping/post_fbl_force_setpoint/'
        fields = ['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]']
        topics['post_fbl_force_setpoint'] = get_data(topic, fields)

        topic = f'/stl_mapping/post_fbl_torque_setpoint/'
        fields = ['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]']
        topics['post_fbl_torque_setpoint'] = get_data(topic, fields)

        topic = f'/{self.robot_ns}/bluerov/disturbance_estimate/'
        fields = ['header/stamp/sec','twist/twist/linear/x','twist/twist/linear/y','twist/twist/linear/z',
                  'twist/twist/angular/x','twist/twist/angular/y','twist/twist/angular/z']
        fields += [f'twist/covariance[{i}]' for i in range(36)]
        topics['disturbance_estimate'] = get_data(topic, fields)
        topics['disturbance_estimate']['covariance'] = np.array([topics['disturbance_estimate'][f'twist/covariance[{i}]'] for i in range(36)]).T

        self.topics = topics

    #! Plotting individual topics

    def plot_images(self, fig: plt.Figure, ax: plt.Axes, images_path: str):
        vehicle_local_position = self.topics['vehicle_local_position']
        loc_pos_time = (vehicle_local_position['timestamp'] - vehicle_local_position['timestamp'][0]) / 1e6
        indices = np.round(np.linspace(0, len(loc_pos_time) - 1, 6)).astype(int)

        # check if folder exists and contains at least one png file
        if not os.path.exists(images_path):
            print(f"Warning: Images path '{images_path}' does not exist.")
            return
        
        # get sorted png files (only first 6)
        image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])[:6]

        rows, cols = 3, 2
        for i, image_file in enumerate(image_files):
            r, c = divmod(i, cols)

            # compute normalized position [left, bottom, width, height] in parent ax
            left = c / cols
            bottom = 1 - (r + 1) / rows
            width = 1 / cols
            height = 1 / rows

            # inset axes at correct location
            ax_img = ax.inset_axes([left, bottom, width, height])

            img = plt.imread(os.path.join(images_path, image_file))
            ax_img.imshow(img)
            ax_img.axis("off")
            ax_img.set_title(f"t: {loc_pos_time[indices[i]]:.1f} s", fontsize=12)

        # hide the parent axes (so only children are visible)
        ax.axis("off")



    # def plot_images(self,fig:plt.Figure, ax:plt.Axes, images_path:str):
    #     vehicle_local_position = self.topics['vehicle_local_position']
    #     loc_pos_time = (vehicle_local_position['timestamp']-vehicle_local_position['timestamp'][0])/1e6
    #     indices = np.round(np.linspace(0, len(loc_pos_time)-1, 6)).astype(int)

    #     # within images_path, find all png files and sort them
    #     image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
    #     image_files.sort()
    #     # then count them 
    #     N = len(image_files)
    #     # and plot them in the axis in a N/2 x 2 grid
    #     for i, image_file in enumerate(image_files):
    #         img = plt.imread(os.path.join(images_path, image_file))
    #         ax_img = fig.add_subplot(N//2, 2, i+1)
    #         ax_img.imshow(img)
    #         ax_img.axis('off')
    #         # title is time rounded to 1 decimal place
    #         ax_img.set_title(f'Time: {loc_pos_time[indices[i]]:.1f} s')

    def plot_position(self,ax:plt.Axes,plot_robot:int=0):
        vehicle_local_position = self.topics['vehicle_local_position']
        vehicle_attitude = self.topics['vehicle_attitude']
        loc_pos_time = (vehicle_local_position['timestamp']-vehicle_local_position['timestamp'][0])/1e6
        att_time = (vehicle_attitude['timestamp']-vehicle_attitude['timestamp'][0])/1e6

        for X in self.Xs:
            X.plot(ax, alpha=0.8)
        ax.plot(self.plan['x'][:,1], self.plan['x'][:,0], 'r--', label=r'$p_{ref}$')
        ax.plot(vehicle_local_position['y'], vehicle_local_position['x'], label=r'$p$')
        if plot_robot > 1:
            # plot the robot image at the position, 'plot_robot' times interpolated
            # along the executed trajectory
            alphas = np.linspace(0, 1, plot_robot+1)[1::]
            x, y = vehicle_local_position['y'], vehicle_local_position['x']
            vehicle_attitude = self.topics['vehicle_attitude']
            q = np.array([vehicle_attitude['q[0]'], vehicle_attitude['q[1]'], vehicle_attitude['q[2]'], vehicle_attitude['q[3]']]).T
            indices = np.round(np.linspace(0, len(x)-1, plot_robot)).astype(int)
            for cnt, i in enumerate(indices):
                x_i, y_i = x[i], y[i]
                q_idx = np.argmin(np.abs(att_time - loc_pos_time[i]))
                Rot = R.from_quat(q[q_idx],scalar_first=True)    
                yaw = Rot.as_euler('xyz', degrees=True)[2]
                im = ax.imshow(self.robot_img, extent=[x_i - 0.25, x_i + 0.25, y_i - 0.25, y_i + 0.25], origin='upper',alpha=alphas[cnt])
                # now transform accordingly
                trans_data = (
                    Affine2D()
                    .rotate_deg_around(x_i, y_i, -yaw)      # shift so that image center is at (x,y)
                    + ax.transData
                )
                im.set_transform(trans_data)
                # set image to top
                im.set_zorder(10)

        # set background color to light blue
        if self.robot_name == 'bluerov':
            ax.set_facecolor("#6ebeff95")
        else:
            ax.set_facecolor("#8896aa94")

        ax.set_title('Position')
        # ax.set_xlabel('y position (m)')
        # ax.set_ylabel('x position (m)')
        ax.axis('equal')
        ax.grid()
        ax.legend()

    def plot_position_time(self,ax):
        vehicle_local_position = self.topics['vehicle_local_position']
        rosbag_time = (vehicle_local_position['timestamp']-vehicle_local_position['timestamp'][0])/1e6
        ax.plot(self.plan['times'], self.plan['x'][:,0], 'r--')#, label=r'$x_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,1], 'g--')#, label=r'$y_{ref}$')
        if self.threeD:
            ax.plot(self.plan['times'], self.plan['x'][:,2], 'b--')#, label=r'$z_{ref}$')
        ax.plot(rosbag_time, vehicle_local_position['x'], 'r-', label=r'$x$')
        ax.plot(rosbag_time, vehicle_local_position['y'], 'g-', label=r'$y$')
        if self.threeD:
            ax.plot(rosbag_time, vehicle_local_position['z'], 'b-', label=r'$z$')

        # find the maximal spatial deviation
        x_ex = np.array([vehicle_local_position['x'], vehicle_local_position['y'], vehicle_local_position['z']])
        x_ref_interp = np.array([np.interp(np.array(rosbag_time), self.plan['times'], self.plan['x'][:,i]) for i in range(3)])
        D = np.abs(x_ex - x_ref_interp)
        max_dev, max_idx, max_dim = -np.inf, 0, 0
        for i, d in enumerate(D.T):
            if np.max(d) > max_dev:
                max_dev = np.max(d)
                max_idx = i
                max_dim = np.argmax(d)
        t = rosbag_time[max_idx]
        # print(f"t: {t}")

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
        vehicle_local_position = self.topics['vehicle_local_position']
        rosbag_time = (vehicle_local_position['timestamp']-vehicle_local_position['timestamp'][0])/1e6

        ax.plot(self.plan['times'], self.plan['x'][:,7], 'r--')#, label=r'$\dot{x}_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,8], 'g--')#, label=r'$\dot{y}_{ref}$')
        if self.threeD:
            ax.plot(self.plan['times'], self.plan['x'][:,9], 'b--')#, label=r'$\dot{z}_{ref}$')
        # TODO: check whether body and inertial frame are mixed up here
        ax.plot(rosbag_time, -vehicle_local_position['vy'], 'r-', label=r'$\dot{x}$')
        ax.plot(rosbag_time, -vehicle_local_position['vx'], 'g-', label=r'$\dot{y}$')
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
        vehicle_attitude = self.topics['vehicle_attitude']
        rosbag_time = (vehicle_attitude['timestamp']-vehicle_attitude['timestamp'][0])/1e6

        q = np.array([vehicle_attitude['q[0]'], vehicle_attitude['q[1]'], vehicle_attitude['q[2]'], vehicle_attitude['q[3]']]).T

        ax.plot(self.plan['times'], np.abs(self.plan['x'][:,3]), 'r--')#, label=r'$q^w_{ref}$')
        ax.plot(self.plan['times'], np.abs(self.plan['x'][:,4]), 'g--')#, label=r'$q^x_{ref}$')
        ax.plot(self.plan['times'], np.abs(self.plan['x'][:,5]), 'b--')#, label=r'$q^y_{ref}$')
        ax.plot(self.plan['times'], np.abs(self.plan['x'][:,6]), 'c--')#, label=r'$q^z_{ref}$')
        ax.plot(rosbag_time, np.abs(q[:,0]), 'r-', label=r'$q^w$')
        ax.plot(rosbag_time, np.abs(q[:,1]), 'g-', label=r'$q^x$')
        ax.plot(rosbag_time, np.abs(q[:,2]), 'b-', label=r'$q^y$')
        ax.plot(rosbag_time, np.abs(q[:,3]), 'c-', label=r'$q^z$')
        ax.set_title('Attitude')
        # ax.set_xlabel('time (s)')
        ax.set_ylabel('quaternion')
        ax.set_ylim([-0.1, 2.0])
        ax.grid()
        # horizontal legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

    def plot_angular_velocity(self,ax):
        vehicle_angular_velocity = self.topics['vehicle_angular_velocity']
        rosbag_time = (vehicle_angular_velocity['timestamp']-vehicle_angular_velocity['timestamp'][0])/1e6

        w = np.array([vehicle_angular_velocity['xyz[0]'], vehicle_angular_velocity['xyz[1]'], vehicle_angular_velocity['xyz[2]']]).T
        if self.threeD:
            ax.plot(self.plan['times'], self.plan['x'][:,10], 'g--')#, label=r'$\theta_{ref}$')
            ax.plot(self.plan['times'], self.plan['x'][:,11], 'r--')#, label=r'$\phi_{ref}$')
        ax.plot(self.plan['times'], self.plan['x'][:,12], 'b--')#, label=r'$\psi_{ref}$')
        if self.threeD:
            ax.plot(rosbag_time, w[:,0], 'r-', label=r'$\phi$')
            ax.plot(rosbag_time, w[:,1], 'g-', label=r'$\theta$')
        ax.plot(rosbag_time, w[:,2], 'b-', label=r'$\psi$')
        ax.set_title('Angular Velocity')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('angular velocity (rad/s)')
        ax.set_ylim([-0.2,0.6])
        ax.grid()
        # ax.legend()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

    def plot_force(self,ax):
        vehicle_thrust_setpoint = self.topics['post_fbl_force_setpoint']
        rosbag_time = (vehicle_thrust_setpoint['timestamp']-vehicle_thrust_setpoint['timestamp'][0])/1e6

        f = np.array([vehicle_thrust_setpoint['xyz[0]'], vehicle_thrust_setpoint['xyz[1]'], vehicle_thrust_setpoint['xyz[2]']]).T
        ax.plot(rosbag_time, f[:,0], 'b-', label=r'$f_x$')
        ax.plot(rosbag_time, f[:,1], 'g-', label=r'$f_y$')
        if self.threeD:
            ax.plot(rosbag_time, f[:,2], 'r-', label=r'$f_z$')
        ax.axhline(self.robot.U.lower_bounds[0], color='k', linestyle='--')
        ax.axhline(self.robot.U.upper_bounds[0], color='k', linestyle='--')
        ax.set_title('Thrust Setpoint')
        # ax.set_xlabel('time (s)')
        ax.set_ylabel('thrust (N)')
        ax.grid()
        ax.legend()

    def plot_disturbance(self,ax,sigma=1):
        disturbance_estimate = self.topics['disturbance_estimate']
        # TODO: ensure time is properly recorded in message
        # rosbag_time = np.array([float(t) for t in disturbance_estimate['time']])
        # rosbag_time = (rosbag_time - rosbag_time[0])
        rosbag_time = (self.topics['post_fbl_force_setpoint']['timestamp']-self.topics['post_fbl_force_setpoint']['timestamp'][0])/1e6
        rosbag_time = np.linspace(rosbag_time[0], rosbag_time[-1], len(disturbance_estimate['twist/twist/linear/x']))

        dx = disturbance_estimate['twist/twist/linear/x']
        dy = disturbance_estimate['twist/twist/linear/y']
        dz = disturbance_estimate['twist/twist/linear/z']
        # cov = disturbance_estimate['twist.covariance']

        cov_x = 3*disturbance_estimate['covariance'][:,0]
        cov_y = 3*disturbance_estimate['covariance'][:,7]
        cov_z = 3*disturbance_estimate['covariance'][:,14]

        ax.plot(rosbag_time, dx, 'b-', label=r'$d_x$')
        ax.fill_between(rosbag_time, dx - sigma*cov_x, dx + sigma*cov_x, color='b', alpha=0.2)
        ax.plot(rosbag_time, dy, 'g-', label=r'$d_y$')
        ax.fill_between(rosbag_time, dy - sigma*cov_y, dy + sigma*cov_y, color='g', alpha=0.2)
        if self.threeD:
            ax.plot(rosbag_time, dz, 'r-', label=r'$d_z$')
            ax.fill_between(rosbag_time, dz - sigma*cov_z, dz + sigma*cov_z, color='r', alpha=0.2)
        # vertical lines of \alpha D
        ax.axhline(self.plan['alpha']*self.robot.D.lower_bounds[0], color='k', linestyle='--')
        ax.axhline(self.plan['alpha']*self.robot.D.upper_bounds[0], color='k', linestyle='--')
        # ax.set_ylim([-10,10])
        ax.set_title('Estimated Disturbance')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('disturbance (N)')
        ax.grid()
        ax.legend()
