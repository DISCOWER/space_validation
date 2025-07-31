#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

import casadi as cs
from acados_template import AcadosModel
from Utilities.smarc_modelling.src.smarc_modelling.vehicles import BlueROV


def bluerov_model_wrench():
    model = AcadosModel()
    model.name = 'bluerov'

    p = cs.MX.sym('p', 3)  # position
    q = cs.MX.sym('q', 4)  # quaternion
    v = cs.MX.sym('v', 3)  # linear velocity
    w = cs.MX.sym('w', 3)  # angular velocity
    x = cs.vertcat(p, q, v, w)

    # controls
    u = cs.MX.sym('u', 6)  # forces and torques

    # reference
    x_ref = cs.MX.sym('x_ref', x.size()[0])
    u_ref = cs.MX.sym('u_ref', u.size()[0])

    # disturbance estimate
    fd = cs.MX.sym('fd', 3)  # force disturbance
    td = cs.MX.sym('td', 3)  # torque disturbance

    model.p = cs.vertcat(x_ref, u_ref, fd, td)

    # Assign dynamics and controls
    bluerov = BlueROV(iX=cs.MX)
    model.f_expl_expr = bluerov.calculate_disturbed_dynamics(x, u, fd, td)
    model.x = x
    model.u = u

    # limits
    model.u_min = bluerov.U.lower_bound
    model.u_max = bluerov.U.upper_bound

    return model


