import numpy as np
from scipy.spatial.transform import Rotation as R

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from Utilities.Robots import FreeFlyer, LinearFreeFlyer6DoF
from Utilities.beziers import eval_bezier

model_name = 'atmos' # Change to 'atmos' or 'bluerov if needed
data = np.load(f'space_validation/Planning/solutions/{model_name}_solution.npz')

x = data['x']
u = data['u']
alpha = data['alpha']
dt = data['dt']

sp_robot = LinearFreeFlyer6DoF()
# pick a few indices where you see the error
indices = [0, 5, 10, 15, 20]   # change to indices relevant to your run
m = sp_robot.mass
I = sp_robot.inertia  # (3,3)

def skew_from_w(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

for i in indices:
    # planned (linear) values you start from
    p_W = x[i, 0:3].copy()            # world position
    quat = x[i, 3:7].copy()           # your stored quat (check order)
    v_W = x[i, 6:9].copy()            # planned world velocity
    w_W = x[i, 9:12].copy()           # planned world angular rate (or body? confirm)
    F_W = u[i, 0:3].copy()            # planned world force
    tau_W = u[i, 3:6].copy()          # planned world torque

    # build rotation (careful with scalar-first vs scalar-last)
    # If your quaternions are scalar-first (w,x,y,z) use: R.from_quat([x,y,z,w])
    # If scalar-last (x,y,z,w) just use R.from_quat(quat)
    # Try both if unsure. We'll attempt scalar-first first:
    try:
        R_d = R.from_quat(quat, scalar_first=True).as_matrix()
        scalar_first_ok = True
    except Exception:
        R_d = R.from_quat(quat).as_matrix()
        scalar_first_ok = False

    # conversions we *expect*
    v_B_expected = R_d.T @ v_W
    w_B_expected = R_d.T @ w_W
    F_B_expected = R_d.T @ F_W
    # alternate (if your code used scipy.apply etc): test that too
    # compute transport term and model accelerations
    w_cross = skew_from_w(w_B_expected)
    vBdot_model = (1.0/m) * F_B_expected - (w_cross @ v_B_expected)        # body-frame RHS
    # world acceleration reconstructed from body-dynamics:
    Rdot_vB = R_d @ (np.cross(w_B_expected, v_B_expected))               # Rdot v_B = R (w x v_B)
    a_W_from_body = R_d @ vBdot_model + Rdot_vB                         # should equal (1/m) * F_W
    a_W_linear = (1.0/m) * F_W

    # differences
    print("=== index", i, "scalar_first_used:", scalar_first_ok, "===")
    print("p_W:", p_W)
    print("quat:", quat)
    print("v_W:", v_W, " -> v_B_expected:", v_B_expected)
    print("w_W:", w_W, " -> w_B_expected:", w_B_expected)
    print("F_W:", F_W, " -> F_B_expected:", F_B_expected)
    print("vBdot_model (body):", vBdot_model)
    print("a_W_from_body:", a_W_from_body)
    print("a_W_linear:", a_W_linear)
    print("a_W_from_body - a_W_linear:", a_W_from_body - a_W_linear)
    print("norm difference:", np.linalg.norm(a_W_from_body - a_W_linear))
    print()
