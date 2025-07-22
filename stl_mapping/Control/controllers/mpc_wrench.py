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

import numpy as np
import os
from scipy.linalg import block_diag
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from ..models.atmos_wrench import atmos_model_wrench

class MpcWrench():
    def __init__(self):
        # Define the controller parameters
        self.dt = 0.2               # MPC time step [s]
        self.Nx = 30                # Prediction horizon, states             
        self.Nu = 30                # Prediction horizon, inputs
        self.Q = np.diag([          # State weighting matrix
            1e0, 1e0, 1e0,
            3e1, 3e1, 3e1, 
            1e2, 
            3e1, 3e1, 3e1])             
        self.R = 0.1*np.diag([          # State weighting matrix
            1e0, 1e0, 1e0,
            1e0, 1e0, 1e0]) 
        self.P = 10 * self.Q        # Terminal state weighting matrix
        
        # Bounds
        self.lbx = np.array([0+0.25, -1.58+0.25, -0.5, -0.5, -3])
        self.ubx = np.array([4.1-0.25, 1.74-0.25, 0.5, 0.5, 3])
        self.idxbx = np.array([0, 1, 3, 4, 12]) # Indexes of states that are bounded

        # Weight on slack varibles
        self.W_slack = np.array([1e4]*len(self.idxbx))
        self.idx_slack = np.array([0, 1, 2, 3, 4]) # Indexes of slack variables

        # Create the OCP
        self.solver = self.setup()

    def setup(self):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        
        # Set directory for code generation and json file
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        codegen_dir = os.path.join(package_root, 'mpc_codegen')
        json_path = os.path.join(codegen_dir, 'acados_ocp.json')
        os.makedirs(codegen_dir, exist_ok=True)
        ocp.code_export_directory = codegen_dir

        # Define the model
        model = atmos_model_wrench()
        ocp.model = model

        # Set dimensions
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        self.nx = nx
        self.nu = nu
        ocp.parameter_values = np.zeros(nx+nu+6)  # x_ref, u_ref, fd, td

        # Get variables
        x_ref = model.p[:nx]
        u_ref = model.p[nx:nx+nu]

        # Set slack variables costs
        ocp.cost.Zl = self.W_slack
        ocp.cost.Zu = self.W_slack
        ocp.cost.zl = np.zeros(self.idx_slack.size)
        ocp.cost.zu = np.zeros(self.idx_slack.size)
        # ocp.cost.Zl_e = ocp.cost.Zl
        # ocp.cost.Zu_e = ocp.cost.Zu
        # ocp.cost.zl_e = ocp.cost.zl
        # ocp.cost.zu_e = ocp.cost.zu

        # Define the cost function
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # Set the weighting matrices
        ocp.cost.W = block_diag(self.Q, self.R)
        ocp.cost.W_e = block_diag(self.P)

        # quat_error = ca.fabs(model.x[6:10].T @ x_ref[6:10])
        quat_error = (model.x[6:10].T @ x_ref[6:10])**2
        # quat_error = ca.fmax(0, ca.fmin(1, quat_error))
        ocp.model.cost_y_expr = ca.vertcat(
            model.x[0:3] - x_ref[0:3],   # Position error
            model.x[3:6] - x_ref[3:6],   # Velocity error
            quat_error,
            model.x[10:13] - x_ref[10:13],  # Angular velocity error
            model.u - u_ref, # Control error
        )
        # Terminal cost 
        ocp.model.cost_y_expr_e = ca.vertcat(
            model.x[0:3] - x_ref[0:3],
            model.x[3:6] - x_ref[3:6],
            quat_error,
            model.x[10:13] - x_ref[10:13],
        )
        ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.shape[0])  # Reference for full cost function
        ocp.cost.yref[6] = 1    # Quaternion reference
        ocp.cost.yref_e = np.zeros(ocp.model.cost_y_expr_e.shape[0])  # Terminal reference
        ocp.cost.yref_e[6] = 1  # Quaternion reference

        # Constraints
        ocp.constraints.lbu = model.u_min
        ocp.constraints.ubu = model.u_max
        ocp.constraints.idxbu = np.arange(nu)

        ocp.constraints.lbx = self.lbx
        ocp.constraints.ubx = self.ubx
        ocp.constraints.idxbx = self.idxbx  # All states are bounded
        ocp.constraints.idxsbx = np.arange(len(self.idxbx)) # All states are slack variables

        ocp.constraints.x0 = np.zeros(nx)  # Initial state
        ocp.constraints.x0[6] = 1  # Initial quaternion

        # Set up the constraints for the other agents
        # ocp.constraints.lh = np.full(0, -1e9)   # lower bounds on con_h_expr
        # ocp.constraints.uh = np.zeros(0)  # no upper bounds (one-sided constraint)  
        # ocp.constraints.idxsh = self.idx_slack  # index of slack variables corresponding to con_h_expr
        # ocp.constraints.lh_e = ocp.constraints.lh
        # ocp.constraints.uh_e = ocp.constraints.uh
        # ocp.constraints.idxsh_e = ocp.constraints.idxsh 

        # Set the prediction horizon
        ocp.solver_options.N_horizon = self.Nx
        ocp.solver_options.tf = self.dt * self.Nx

        # Set the solver options
        useRTI = True
        if useRTI:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
            ocp.solver_options.nlp_solver_max_iter = 1
        else:
            ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM,
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.print_level = 0

        ocp_solver = AcadosOcpSolver(ocp, json_file=json_path)
        return ocp_solver

    def get_input(self, x0, x_ref, fd=np.zeros(3), td=np.zeros(3)):
        # Properly set x0, i.e. constrain it to x0
        print("trying to set initial state")
        self.solver.set(0, "lbx", x0.flatten())
        self.solver.set(0, "ubx", x0.flatten())

        # Update reference
        if x_ref.ndim == 1:
            x_ref = x_ref.reshape(-1, 1)
        len_x_ref = x_ref.shape[1]
        for k in range(self.Nx + 1):
            if k < len_x_ref:
                x_ref_k = x_ref[:self.nx,k]
                u_ref_k = x_ref[self.nx:self.nx+self.nu,k]
            else:
                x_ref_k = x_ref[:self.nx,-1]
                u_ref_k = x_ref[self.nx:self.nx+self.nu,-1]
            self.solver.set(k, "p", np.concatenate((x_ref_k, u_ref_k, fd, td), axis=0))

        status = self.solver.solve()
        u_opt = self.solver.get(0, 'u')
        if status != 0:
            print("Solver failed. Retrying with warm-start reset.")
            self.solver.reset()
            status = self.solver.solve()
            if status != 0:
                print("Solver failed again. Using previous solution.")
                u_opt =  self.solver.get(1, 'u')
        
        # get solution
        x_pred = np.ndarray((self.Nx+1, self.nx))
        for i in range(self.Nx):
            x_pred[i,:] = self.solver.get(i, "x")
        x_pred[self.Nx,:] = self.solver.get(self.Nx, "x")
        
        return u_opt, x_pred