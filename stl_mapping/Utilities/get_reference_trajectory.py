import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from Utilities.beziers import value_bezier
from Utilities.rotations import quat_to_euler_np, euler_to_quat_np

class ReferenceTrajectory:
    def __init__(self, r, dr, q, dt):
        self.r = r
        self.dr = dr
        self.q = q
        self.dt = dt
        self.N = r.shape[0]

def get_reference_trajectory(t:float, trajectory:ReferenceTrajectory, order:str='zyx'):
    idx = min(int(t / trajectory.dt), trajectory.N-1)
    s = min(1,(t - idx * trajectory.dt) / trajectory.dt)
    # print(f"idx: {idx}, s: {s}")

    p = value_bezier(trajectory.r[idx], s)
    v = value_bezier(trajectory.dr[idx], s)

    slerp = Slerp([0, 1], Rotation.from_quat(trajectory.q[idx], scalar_first=True))
    q_val = slerp(s).as_quat(scalar_first=True)

    ds = 1e-4
    dq_val = slerp([s, min(1,s+ds)]).as_euler(order)
    dq_val = (dq_val[1] - dq_val[0]) / (ds*trajectory.dt)
    # dq_val = np.zeros((3,))   # This for slow-moving tests

    # print(f"q_val: {quat_to_euler_np(q_val, order=order)}")
    # print(f"dq_val: {dq_val}")
    return np.concatenate((p, q_val, v, dq_val))


if __name__ == "__main__":

    data = np.load('stl_mapping/Planning/solutions/sp_solution_quat.npz')
    x_ff = data['x_ff']
    u_ff = data['u_ff']
    dt = data['dt']

    data = np.load('stl_mapping/Planning/solutions/sp_solution_bezier.npz')
    r = data['r']
    dr = data['dr']
    ddr = data['ddr']
    q = data['q']
    trajectory = ReferenceTrajectory(r, dr, q, dt)

    for t in np.arange(0, dt*(x_ff.shape[0]-1)+1.0, 0.1):
        x_ref = get_reference_trajectory(t, trajectory)
        print(f"t: {t}, x_ref: {x_ref}")