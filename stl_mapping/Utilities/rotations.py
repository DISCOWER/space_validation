
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
    rot_mat = q_to_rot_mat(q)

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

def quat_x_to_euler_x_cs(x):
    # assumes x = [p, q, dp, dq]
    euler_x = cs.vertcat(
        x[0:3],                   # position
        quat_to_euler_cs(x[3:7]),    # euler angles
        x[7:10],                  # linear velocity
        x[10:13]                  # angular velocity
    )
    return euler_x

def euler_x_to_quat_x_cs(x):
    # assumes x = [p, q, dp, dq]
    p, q, dp, dq = x[0:3], x[3:6], x[6:9], x[9:12]
    quat_x = cs.vertcat(
        p,                   # position
        euler_to_quat_cs(q),    # quaternion
        dp,                  # linear velocity
        dq                   # angular velocity
    )
    return quat_x

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