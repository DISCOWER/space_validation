
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

def quat_mult(q1, q2):
    return cs.vertcat(
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
            q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
            q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        )

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

def quat_to_euler_np(q,order='zyz'):
    """
    Convert quaternion (qw, qx, qy, qz) to Euler angles (roll, pitch, yaw).
    Angles are in radians.
    """
    q = R.from_quat(q, scalar_first=True)  # convert to scipy Rotation object
    euler = q.as_euler(order, degrees=False)  # returns (
    return euler

def quat_x_to_euler_x_np(x, order='zyz'):
    # assumes x = [p, q, dp, dq]
    euler_x = np.concatenate((
        x[0:3],                   # position
        quat_to_euler_np(x[3:7], order=order),    # euler angles
        x[7:10],                  # linear velocity
        x[10:13]                  # angular velocity
    ))
    return euler_x


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

def ned_to_enu(pos_ned, quat_ned):
    """
    Convert position and quaternion from NED to ENU frame.
    
    Parameters:
        pos_ned (np.ndarray): Position in NED frame.
        quat_ned (np.ndarray): Quaternion in NED frame.
    
    Returns:
        tuple: Position and quaternion in ENU frame.
    """
    T = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    pos_enu = T @ pos_ned
    r_ned = R.from_quat(quat_ned)
    r_enu = T @ r_ned.as_matrix() @ T.T
    quat_enu = R.from_matrix(r_enu).as_quat()
    return pos_enu, quat_enu

def enu_to_ned(pos_enu, quat_enu):
    """
    Convert position and quaternion from ENU to NED frame.
    
    Parameters:
        pos_enu (np.ndarray): Position in ENU frame.
        quat_enu (np.ndarray): Quaternion in ENU frame.
    
    Returns:
        tuple: Position and quaternion in NED frame.
    """
    T = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    pos_ned = T @ pos_enu
    r_enu = R.from_quat(quat_enu, scalar_first=True)
    r_ned = T @ r_enu.as_matrix() @ T.T
    quat_ned = R.from_matrix(r_ned).as_quat(scalar_first=True)
    return pos_ned, quat_ned

def u_enu_to_ned(u):
    """
    Convert control inputs from ENU to NED frame.
    
    Parameters:
        u (np.ndarray): Control inputs in ENU frame.
    
    Returns:
        np.ndarray: Control inputs in NED frame.
    """
    T = np.diag([1, -1, -1, 1, -1, -1])
    u_ned = T @ u
    return u_ned