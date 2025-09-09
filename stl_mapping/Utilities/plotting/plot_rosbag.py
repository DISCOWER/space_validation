#!/usr/bin/env python3
import os
from pathlib import Path
import scienceplots
from matplotlib import rc
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

import numpy as np
import matplotlib.pyplot as plt
from rosbag_class import RosBagClass
import pandas as pd

robot_name = 'bluerov'
experiment = 1

# 2D experiment bluerov
folder_path = f"{str(Path.home())}/space_ws/rosbags/exp_1_uw_2/rosbag2_2025_09_09-12_25_24/"
offset=np.array([0.5,0,0])
threeD=False

# # 3D experiment bluerov
# folder_path = f"{str(Path.home())}/space_ws/rosbags/exp_2_uw_1/rosbag2_2025_09_09-17_12_26/"
# offset = np.array([0.3,0.3,1.6])
# threeD=True

csv_file_path = os.path.join(folder_path, 'plotjuggler.csv')
csv_file  = pd.read_csv(csv_file_path)

plan_file_path = os.path.join(folder_path, f'plans/{robot_name}_nonlinear_solution.npz')
plan = np.load(plan_file_path)

images_path = os.path.join(folder_path, 'images')

obj = RosBagClass(csv_file=csv_file,
                  plan=plan,
                  robot_name=robot_name,
                  threeD=threeD,
                  offset=offset,
                  exp=experiment)

# Set up plot style
plt.style.use(['science'])
rc('text', usetex=True)
rc('font', family='times', size=12)

plt.rcParams['legend.frameon'] = True             # Enable legend frame
plt.rcParams['legend.facecolor'] = 'white'        # Set background color
plt.rcParams['legend.edgecolor'] = 'white'        # Set border color
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc'] = 'best'

#! narrow images
# fig = plt.figure(figsize=(15, 4.5),constrained_layout=True)
# gs = fig.add_gridspec(2,5, figure=fig)
# ax_img = fig.add_subplot(gs[:,0])
# ax_p = fig.add_subplot(gs[:,1])
# ax_p2 = fig.add_subplot(gs[0,2])
# ax_v = fig.add_subplot(gs[1,2])
# ax_q = fig.add_subplot(gs[0,3])
# ax_w = fig.add_subplot(gs[1,3])
# ax_f = fig.add_subplot(gs[0,4])
# ax_d = fig.add_subplot(gs[1,4])

# obj.plot_images(fig, ax_img, images_path)
# obj.plot_position(ax_p,plot_robot=6)
# obj.plot_position_time(ax_p2)
# obj.plot_velocity(ax_v)
# obj.plot_attitude(ax_q)
# obj.plot_angular_velocity(ax_w)
# obj.plot_force(ax_f)
# obj.plot_disturbance(ax_d,sigma=2)

#! wide images
fig = plt.figure(figsize=(15, 4.5),constrained_layout=True)
gs = fig.add_gridspec(2,5, figure=fig)
ax_img = fig.add_subplot(gs[:,0:2])
ax_p = fig.add_subplot(gs[:,2])
ax_p2 = fig.add_subplot(gs[0,3])
ax_q = fig.add_subplot(gs[1,3])
ax_f = fig.add_subplot(gs[0,4])
ax_d = fig.add_subplot(gs[1,4])

obj.plot_images(fig, ax_img, images_path)
obj.plot_position(ax_p,plot_robot=6)
obj.plot_position_time(ax_p2)
obj.plot_attitude(ax_q)
obj.plot_force(ax_f)
obj.plot_disturbance(ax_d,sigma=2)


# plt.show()
plt.savefig(f'stl_mapping/Utilities/plotting/figures/{robot_name}_{experiment}.pdf')
