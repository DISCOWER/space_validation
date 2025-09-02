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
import os, sys
from scipy.linalg import block_diag
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from models.atmos_wrench import atmos_model_wrench
from models.bluerov_wrench import bluerov_model_wrench
from rclpy.node import Node

from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint
from Utilities.ros.qos_profiles import NORMAL_QOS

import rclpy
rclpy.init()

class MpcWrench(Node):
    def __init__(self, model_name:str='atmos'):
        super().__init__('mpc_wrench')
        # Create two publishers for pre- and post-feedback equivalence control
        self.publisher_force_control = self.create_publisher(
            VehicleThrustSetpoint,
            '/stl_mapping/force_setpoint',
            NORMAL_QOS
        )
        self.publisher_torque_control = self.create_publisher(
            VehicleTorqueSetpoint,
            '/stl_mapping/torque_setpoint',
            NORMAL_QOS
        )

        # Define the controller parameters
        self.dt = 0.1               # MPC time step [s]
        self.Nx = 30                # Prediction horizon, states             
        self.Nu = 30                # Prediction horizon, inputs

        #! ATMOS weights
        if model_name == 'atmos':
            self.Q = np.diag([          # State weighting matrix
                1e2, 1e2, 1e2,
                5e1, 5e1, 5e1, 5e1,
                3e1, 3e1, 3e1,  
                3e1, 3e1, 3e1])             
            self.R = 0.1*np.diag([          # State weighting matrix
                1e0, 1e0, 1e0,
                1e0, 1e0, 1e0]) 
            self.P = 10 * self.Q        # Terminal state weighting matrix
            #! ATMOS Bounds
            self.lbx = np.array([0+0.25, -1.58+0.25, -0.5, -0.5, -3])
            self.ubx = np.array([4.1-0.25, 1.74-0.25, 0.5, 0.5, 3])
            self.idxbx = np.array([0, 1, 7, 8, 12]) # Indexes of states that are bounded
        elif model_name == 'bluerov':
            #! BlueROV weights
            self.Q = np.diag([          # State weighting matrix
                5e2, 5e2, 5e2,
                1e4, 1e4, 1e4, 1e4,
                3e0, 3e0, 3e0,  
                3e3, 3e3, 3e3])          
            self.R = 0.1*np.diag([          # State weighting matrix
                1e0, 1e0, 1e0,
                1e1, 1e1, 1e1]) 
            self.P = 2 * self.Q        # Terminal state weighting matrix
            # #! Sys-id weights
            # self.Q = np.diag([          # State weighting matrix
            #     1e1, 1e1, 1e1,
            #     5e3, 5e3, 5e3, 5e3,
            #     3e0, 3e0, 3e0,  
            #     3e0, 3e0, 3e0])          
            # self.R = 0.1*np.diag([          # State weighting matrix
            #     1e0, 1e0, 1e0,
            #     1e0, 1e0, 1e0]) 
            # self.P = 10 * self.Q        # Terminal state weighting matrix
            # self.Q = np.diag([          # State weighting matrix
            #     1e0, 1e0, 1e0,
            #     6e2, 3e2, 3e2, 3e2,
            #     3e1, 3e1, 3e1,  
            #     1e1, 1e1, 1e1])          
            # self.R = np.diag([          # State weighting matrix
            #     1e-2, 1e-2, 1e-2,
            #     1e2, 1e2, 1e2]) 
            # self.P = 20 * self.Q        # Terminal state weighting matrix

            #! BlueROV Bounds
            self.lbx = np.array([0.5, -2, 0,  -1, -1, -1])
            self.ubx = np.array([8.5, 2, 2.5, 1,  1,  1])
            self.idxbx = np.array([0, 1, 2, 7, 8, 9]) # Indexes of states that are bounded
        else:
            raise ValueError(f"Model {model_name} not recognized.")

        # Weight on slack varibles
        self.W_slack = np.array([1e4]*len(self.idxbx))
        self.idx_slack = np.array([0, 1, 2, 3, 4, 5]) # Indexes of slack variables

        # Create the OCP
        self.model_name = model_name
        self.solver = self.setup()

    def publish_messages(self, u_opt):
        # Publish pre-feedback control messages
        thrust_msg = VehicleThrustSetpoint()
        thrust_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        thrust_msg.xyz = [u_opt[0], u_opt[1], u_opt[2]]
        self.publisher_force_control.publish(thrust_msg)

        torque_msg = VehicleTorqueSetpoint()
        torque_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        torque_msg.xyz = [u_opt[3], u_opt[4], u_opt[5]]
        self.publisher_torque_control.publish(torque_msg)

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
        if self.model_name == 'atmos':
            print("Using Atmos model")
            model = atmos_model_wrench()
        elif self.model_name == 'bluerov':
            print("Using BlueROV model")
            model = bluerov_model_wrench()
        else:
            raise ValueError(f"Model {self.model_name} not recognized. Use 'atmos' or 'bluerov'.")
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

        q1 = x_ref[3:7]
        q2 = model.x[3:7]
        # Sice unit quaternion, quaternion inverse is equal to its conjugate
        q_conj = ca.vertcat(q2[0], -q2[1], -q2[2], -q2[3])
        q2 = q_conj/ca.norm_2(q2)
        
        # q_error = q1 @ q2^-1
        q_w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        q_x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        q_y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        q_z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

        q_error = ca.vertcat(q_w, q_x, q_y, q_z)
        # q_error = ca.if_else(q_w < 0, -q_error, q_error)
        q_error = q_error * ca.sign(q_w)

        #! old
        # q_error = ca.fabs(model.x[6:10].T @ x_ref[6:10])
        # q_error = (model.x[3:7].T @ x_ref[3:7])**2

        ocp.model.cost_y_expr = ca.vertcat(
            model.x[0:3] - x_ref[0:3],   # Position error
            q_error,
            model.x[7:10] - x_ref[7:10],   # Velocity error
            model.x[10:13] - x_ref[10:13],  # Angular velocity error
            model.u - u_ref, # Control error
        )
        # Terminal cost 
        ocp.model.cost_y_expr_e = ca.vertcat(
            model.x[0:3] - x_ref[0:3],
            q_error,
            model.x[7:10] - x_ref[7:10],
            model.x[10:13] - x_ref[10:13],
        )
        ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.shape[0])  # Reference for full cost function
        ocp.cost.yref[3] = 1    # Quaternion reference
        ocp.cost.yref_e = np.zeros(ocp.model.cost_y_expr_e.shape[0])  # Terminal reference
        ocp.cost.yref_e[3] = 1  # Quaternion reference

        # Constraints
        ocp.constraints.lbu = model.u_min
        ocp.constraints.ubu = model.u_max
        ocp.constraints.idxbu = np.arange(nu)

        ocp.constraints.lbx = self.lbx
        ocp.constraints.ubx = self.ubx
        ocp.constraints.idxbx = self.idxbx  # All states are bounded
        ocp.constraints.idxsbx = np.arange(len(self.idxbx)) # All states are slack variables

        ocp.constraints.x0 = np.zeros(nx)  # Initial state
        ocp.constraints.x0[3] = 1  # Initial quaternion

        # Set the prediction horizon
        ocp.solver_options.N_horizon = self.Nx
        ocp.solver_options.tf = self.dt * self.Nx

        # Set the solver options
        # These settings work on both bluerov and atmos model
        useRTI = True
        if useRTI:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
            ocp.solver_options.nlp_solver_max_iter = 1
        else:
            ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM,
        ocp.solver_options.hpipm_mode = 'ROBUST'
        ocp.solver_options.integrator_type = 'ERK' # 'ERK', 'DISCRETE', 'IRK'
        ocp.solver_options.sim_method_newton_iter = 2
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'

        ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
        ocp.solver_options.qp_tol = 1e-6       # QP tolerance
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        ocp.solver_options.regularize_method = 'NO_REGULARIZE'

        ocp.solver_options.print_level = 0

        ocp_solver = AcadosOcpSolver(ocp, json_file=json_path)
        return ocp_solver

    def get_input(self, x0, x_ref, fd=np.zeros(3), td=np.zeros(3)):
        # Properly set x0, i.e. constrain it to x0
        self.solver.set(0, "lbx", x0.flatten())
        self.solver.set(0, "ubx", x0.flatten())

        # self.get_logger().info("\n\n")
        # self.get_logger().info(f"x0: {x0[0:7].flatten()}")
        # self.get_logger().info(f"xref: {x_ref[0:7,0].flatten()}")

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
            self.get_logger().info(f"Solver failed: {status}")
            self.solver.reset()
            status = self.solver.solve()
            if status != 0:
                self.get_logger().info("Solver failed again. Using previous solution.")
                u_opt =  self.solver.get(1, 'u')
        
        # get solution
        x_pred = np.ndarray((self.Nx+1, self.nx))
        for i in range(self.Nx):
            x_pred[i,:] = self.solver.get(i, "x")
        x_pred[self.Nx,:] = self.solver.get(self.Nx, "x")

        # self.get_logger().info(f"Cost: {self.solver.get_cost()}")

        # publish the same pre and post fbl because we don't do any fbl
        # but we keep things consistent for later analysis
        self.publish_messages(u_opt)

        return u_opt, x_pred