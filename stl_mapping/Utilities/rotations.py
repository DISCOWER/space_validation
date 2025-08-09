
import casadi as cs
import numpy as np
from scipy.spatial.transform import Rotation as R

def skew_symmetric_cs(v):
    return cs.vertcat(cs.horzcat(0, -v[0], -v[1], -v[2]),
                      cs.horzcat(v[0], 0, v[2], -v[1]),
                      cs.horzcat(v[1], -v[2], 0, v[0]),
                      cs.horzcat(v[2], v[1], -v[0], 0))

def skew_symmetric_np(v):
    return np.array([[0, -v[0], -v[1], -v[2]],
                     [v[0], 0, v[2], -v[1]],
                     [v[1], -v[2], 0, v[0]],
                     [v[2], v[1], -v[0], 0]])

def q_to_rot_mat_cs(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    rot_mat = cs.vertcat(
        cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
        cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))
    return rot_mat

def q_to_rot_mat_np(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])
    return rot_mat


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat_cs(q)
    return cs.mtimes(rot_mat, v)

def euler_to_quat_cs(pry,order='zyz'):
    #TODO: check this function and if order matters
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (qw, qx, qy, qz).
    Angles are in radians.
    """
    pitch, roll, yaw = pry[0], pry[1], pry[2]
    cy = cs.cos(yaw * 0.5)
    sy = cs.sin(yaw * 0.5)
    cr = cs.cos(roll * 0.5)
    sr = cs.sin(roll * 0.5)
    cp = cs.cos(pitch * 0.5)
    sp = cs.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = sy * cr * cp + cy * sr * sp
    qz = sy * sr * cp - cy * cr * sp

    return cs.vertcat(qw, qx, qy, qz)  # return as (qw, qx, qy, qz)

def euler_to_quat_np(pry, order='zyz'):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (qw, qx, qy, qz).
    Angles are in radians.
    """
    r = R.from_euler(order, pry, degrees=False)
    q = r.as_quat(scalar_first=True)  # returns (qw, qx, qy, qz)
    return q

def quat_to_euler_np(q, order='zyz'):
    """
    Convert quaternion (qw, qx, qy, qz) to Euler angles (roll, pitch, yaw).
    Angles are in radians.
    """
    r = R.from_quat(q)
    euler_angles = r.as_euler(order, degrees=False)  # returns (roll, pitch, yaw)
    return euler_angles

def enu_to_flu(w, q):
    """
    Convert angular velocity from ENU to FLU frame.
    
    Parameters:
        w (np.ndarray): Angular velocity in ENU frame.
        q (np.ndarray): Quaternion representing the orientation of FLU in ENU frame.
    
    Returns:
        np.ndarray: Angular velocity in FLU frame.
    """
    r = R.from_quat(q, scalar_first=True)
    w_flu = r.inv().apply(w)
    return w_flu

def quat_to_euler_cs(q):
    """
    Convert quaternion (qw, qx, qy, qz) to Euler angles (roll, pitch, yaw).
    Angles are in radians.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    roll = cs.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    pitch = cs.asin(2 * (qw * qy - qz * qx))
    yaw = cs.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))

    return cs.vertcat(roll, pitch, yaw)

def euler_x_to_quat_x_np(x,order='zyx'):
    # assumes x = [p, q, dp, dq]
    p, q, dp, dq = x[0:3], x[3:6], x[6:9], x[9:12]
    quat_x = np.concatenate((
        p,                   # position
        euler_to_quat_np(q, order=order),    # quaternion
        dp,                  # linear velocity
        dq                   # angular velocity
    ))
    return quat_x

def x_ff_to_x_uw(x_ff, normalize_quat=True, q_rel=None):
    #! Convert [p: ENU, q:FLU->ENU, v:ENU, w:FLU] to [p: NED, q:FRD->NED, v:FRD, w:FRD]
    p, q, v, w = x_ff[0:3], x_ff[3:7], x_ff[7:10], x_ff[10:13]
    p_ned = np.array([p[1], p[0], -p[2]])
    q_ned = 1/np.sqrt(2) * np.array([q[0] + q[3], q[1] + q[2], q[1] - q[2], q[0] - q[3]])
    if normalize_quat:
        q_ned = q_ned / np.linalg.norm(q_ned)  # normalize quaternion
    if q_rel is not None:
        v_flu = R.from_quat(q_rel, scalar_first=True).inv().apply(v)
    else:
        v_flu = R.from_quat(q, scalar_first=True).inv().apply(v)
    v_frd = np.array([v_flu[0], -v_flu[1], -v_flu[2]])
    w_frd = np.array([w[0], -w[1], -w[2]])
    x_uw = np.concatenate((p_ned, q_ned, v_frd, w_frd))
    return x_uw

def x_uw_to_x_ff(x_uw, normalize_quat=True, q_rel=None):
    #! Convert [p: NED, q:FRD->NED, v:FRD, w:FRD] to [p: ENU, q:FLU->ENU, v:ENU, w:FLU]
    p, q, v, w = x_uw[:3], x_uw[3:7], x_uw[7:10], x_uw[10:13]
    p_enu = np.array([p[1], p[0], -p[2]])
    q_enu = 1/np.sqrt(2) * np.array([q[0] + q[3], q[1] + q[2], q[1] - q[2], q[0] - q[3]])
    if normalize_quat:
        q_enu = q_enu / np.linalg.norm(q_enu)  # normalize quaternion
    if q_rel is not None:
        v_ned = R.from_quat(q_rel, scalar_first=True).apply(v)
    else:
        v_ned = R.from_quat(q, scalar_first=True).apply(v)
    v_enu = np.array([v_ned[1], v_ned[0], -v_ned[2]])
    w_enu = np.array([w[0], -w[1], -w[2]])
    x_ff = np.concatenate((p_enu, q_enu, v_enu, w_enu))
    return x_ff

def x_ff_to_x_uw_cs(x_ff, normalize_quat=True, q_rel=None):
    #TODO: q should be inverse here, but how to do that for quaternion?
    p, q, v, w = x_ff[0:3], x_ff[3:7], x_ff[7:10], x_ff[10:13]
    p_ned = cs.vertcat(p[1], p[0], -p[2])
    q_ned = 1/cs.sqrt(2) * cs.vertcat(q[0] + q[3], q[1] + q[2], q[1] - q[2], q[0] - q[3])
    if normalize_quat:
        q_ned = q_ned / cs.norm_2(q_ned)  # normalize quaternion
    v_flu = v_dot_q(v, q)
    v_frd = cs.vertcat(v_flu[0], -v_flu[1], -v_flu[2])
    w_frd = cs.vertcat(w[0], -w[1], -w[2])
    x_uw = cs.vertcat(p_ned, q_ned, v_frd, w_frd)
    return x_uw

def x_uw_to_x_ff_cs(x_uw, normalize_quat=True, q_rel=None):
    p, q, v, w = x_uw[:3], x_uw[3:7], x_uw[7:10], x_uw[10:13]
    p_enu = cs.vertcat(p[1], p[0], -p[2])
    q_enu = 1/cs.sqrt(2) * cs.vertcat(q[0] + q[3], q[1] + q[2], q[1] - q[2], q[0] - q[3])
    if normalize_quat:
        q_enu = q_enu / cs.norm_2(q_enu)  # normalize quaternion
    v_ned = v_dot_q(v, cs.inv(q))
    v_enu = cs.vertcat(v_ned[1], v_ned[0], -v_ned[2])
    w_enu = cs.vertcat(w[0], -w[1], -w[2])
    x_ff = cs.vertcat(p_enu, q_enu, v_enu, w_enu)
    return x_ff

def quat_mult(q1, q2):
    return cs.vertcat(
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
            q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
            q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        )