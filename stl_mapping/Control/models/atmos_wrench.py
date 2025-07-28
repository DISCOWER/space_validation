#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Elias Krantz"
__contact__ = "eliaskra@kth.se"

import casadi as ca
import numpy as np
from acados_template import AcadosModel

def get_rotMat(q):
    rotMat = ca.vertcat(
            ca.horzcat(1-2*q[2]**2-2*q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])),
            ca.horzcat(2*(q[1]*q[2]+q[0]*q[3]), 1-2*q[1]**2-2*q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])),
            ca.horzcat(2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*q[1]**2-2*q[2]**2)
    )
    return rotMat

def quat_mult(q1, q2):
    return ca.vertcat(
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
            q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
            q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        )

def quat_derivative(q, w):
    return 0.5 * quat_mult(q, ca.vertcat(0, w))

def atmos_model_wrench():
    model = AcadosModel()
    model.name = 'atmos_ff'

    # parameters
    mass = 16.8 #(16.8: empty, 17.8: full)  # kg
    mass_inv = 1/mass
    inertia = np.diag([0.315]*3)
    inertia_inv = np.linalg.inv(inertia)

    # states
    p = ca.MX.sym('p', 3)
    q = ca.MX.sym('q', 4)
    v = ca.MX.sym('v', 3)
    w = ca.MX.sym('w', 3)
    x = ca.vertcat(p, q, v, w)

    # controls
    u = ca.MX.sym('u', 6)

    # reference
    x_ref = ca.MX.sym('x_ref', x.size()[0])
    u_ref = ca.MX.sym('u_ref', u.size()[0])

    # disturbance estimate
    fd = ca.MX.sym('fd', 3)  # force disturbance
    td = ca.MX.sym('td', 3)  # torque disturbance

    model.p = ca.vertcat(x_ref, u_ref, fd, td)
    
    rotMat = get_rotMat(q)
    w_cross = ca.vertcat(
        ca.horzcat(0, -w[2], w[1]),
        ca.horzcat(w[2], 0, -w[0]),
        ca.horzcat(-w[1], w[0], 0)
    )

    F = ca.vertcat(u[0], u[1], u[2])
    T = ca.vertcat(u[3], u[4], u[5])

    pdot = v
    qdot = quat_derivative(q, w)
    vdot = mass_inv * ca.mtimes(rotMat, F) + fd * mass_inv
    wdot = ca.mtimes(inertia_inv, (T + td - ca.mtimes(w_cross, ca.mtimes(inertia, w))))

    xdot = ca.vertcat(pdot, qdot, vdot, wdot)

    # Assign dynamics and controls
    model.f_expl_expr = xdot
    model.x = x
    model.u = u

    # limits
    F_lim = 2 * 1.4 * 2/3
    T_lim = 4 * 0.12 * 1.4 * 1/3
    model.u_min = np.array([-F_lim, -F_lim, -F_lim, -T_lim, -T_lim, -T_lim])
    model.u_max = np.array([F_lim, F_lim, F_lim, T_lim, T_lim, T_lim])

    return model