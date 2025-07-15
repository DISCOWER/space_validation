import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from Utilities.beziers import value_bezier

data = np.load('Planning/solutions/sp_solution_quat.npz')

x_ff = data['x_ff']
u_ff = data['u_ff']
dt = data['dt']

data = np.load('Planning/solutions/sp_solution_bezier.npz')
r = data['r']
dr = data['dr']
ddr = data['ddr']
q = data['q']


def get_reference_trajectory(t:float):
    idx = min(int(t / dt), r.shape[0]-1)
    s = (t - idx * dt) / dt
    print(f"idx: {idx}, s: {s}")

    p = value_bezier(r[idx], s)
    v = value_bezier(dr[idx], s)

    slerp = Slerp([0, 1], Rotation.from_quat(q[idx], scalar_first=True))
    q_val = slerp(s).as_quat(scalar_first=True)

    # TODO: this seems to map [1,0,0,0] quat to [pi,0,0] euler angles (should be [0,0,0])
    ds = 1e-6
    dq_val = slerp([s, min(1,s+ds)]).as_euler('xyz')
    dq_val = (dq_val[1] - dq_val[0]) / ds

    print(f"q_val: {q_val}")
    print(f"dq_val: {dq_val}")
    return np.concatenate((p, q_val, v, dq_val))


for t in np.arange(0, dt*(x_ff.shape[0]-1), 0.1):
    print(t)
    x_ref = get_reference_trajectory(t)
    print(f"t: {t}, x_ref: {x_ref}")