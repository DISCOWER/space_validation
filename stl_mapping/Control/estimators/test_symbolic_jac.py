import casadi as cs
import numpy as np

import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)
from Utilities.Robots import FreeFlyer
from Utilities.smarc_modelling.src.smarc_modelling.vehicles.BlueROV import BlueROV  

# robot = FreeFlyer()
robot = BlueROV()

x = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0],dtype=float)
u = np.array([0,0,0,0,0,0],dtype=float)
fd = np.array([0,0,0],dtype=float)
td = np.array([0,0,0],dtype=float)

dx = robot.calculate_disturbed_dynamics(x,u,fd,td)
print(f"dx: {dx}")

x = cs.SX.sym('x', 13)
u = cs.SX.sym('u', 6)
fd = cs.SX.sym('fd', 3)
td = cs.SX.sym('td', 3)
dt = cs.SX.sym('dt', 1)

dx = robot.calculate_disturbed_dynamics(x,u,fd,td)

x_kp1 = x + dx*dt

F = cs.SX.eye(12)
F[0:6,0:6] = cs.jacobian(x_kp1[7:13], x[7:13])
F[0:6,6:12] = cs.jacobian(x_kp1[7:13], cs.vertcat(fd,td))

func = cs.Function('F', [x, u, fd, td, dt], [F])


test = func(np.array([0,0,0,1,0,0,0,1,1,1,2,2,2],dtype=float),
            np.array([0,0,0,0,0,0],dtype=float),
            np.array([0,0,0],dtype=float),
            np.array([0,0,0],dtype=float),
            0.1)

indices = slice(0,3)
print(f"\n\nindices: {indices}")
print(f"test: {np.array(test[indices,0:3])}")
print(f"test: {np.array(test[indices,3:6])}")
print(f"test: {np.array(test[indices,6:9])}")
print(f"test: {np.array(test[indices,9:12])}")

indices = slice(3,6)
print(f"\n\nindices: {indices}")
print(f"test: {np.array(test[indices,0:3])}")
print(f"test: {np.array(test[indices,3:6])}")
print(f"test: {np.array(test[indices,6:9])}")
print(f"test: {np.array(test[indices,9:12])}")

indices = slice(6,9)
print(f"\n\nindices: {indices}")
print(f"test: {np.array(test[indices,0:3])}")
print(f"test: {np.array(test[indices,3:6])}")
print(f"test: {np.array(test[indices,6:9])}")
print(f"test: {np.array(test[indices,9:12])}")

indices = slice(9,12)
print(f"\n\nindices: {indices}")
print(f"test: {np.array(test[indices,0:3])}")
print(f"test: {np.array(test[indices,3:6])}")
print(f"test: {np.array(test[indices,6:9])}")
print(f"test: {np.array(test[indices,9:12])}")

# print(f"dx_sym: {dx}")

# print(f"\n\njac: {cs.jacobian(x_kp1[7:10], x[7:10])}")
# print(f"\n\njac: {cs.jacobian(x_kp1[10:13], x[10:13])}")

# print(f"\n\njac: {cs.jacobian(x_kp1, fd)}")
# print(f"\n\njac: {cs.jacobian(x_kp1, td)}")

# print(f"\n\n cx: {robot.calculate_cx(x)}")

