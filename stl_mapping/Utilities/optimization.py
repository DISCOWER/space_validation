import numpy as np
import casadi as cs
from Utilities.smarc_modelling.src.smarc_modelling.lib.gnc import *
from Utilities.smarc_modelling.src.smarc_modelling.lib.gnc_casadi import *

math_backend = {
    'np': {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'sqrt': np.sqrt,
        'atan2': np.arctan2,
        'abs': np.abs,

        'array': np.array,
        'zeros': np.zeros,
        'diag': np.diag,
        'matmul': np.matmul,
        'concatenate': np.concatenate,
        'norm': np.linalg.norm,
        'inv': np.linalg.inv,
        'pinv': np.linalg.pinv,

        'skew_symmetric': skew_symmetric,
        'm2c': m2c,
        'gvect': gvect,
        'calculate_dcm': calculate_dcm,
        'quaternion_to_dcm': quaternion_to_dcm,
        'quaternion_to_angles': quaternion_to_angles,
    },
    'cs': {
        'sin': cs.sin,
        'cos': cs.cos,
        'tan': cs.tan,
        'exp': cs.exp,
        'sqrt': cs.sqrt,
        'atan2': cs.atan2,
        'abs': cs.fabs,

        'array': lambda x: cs.vertcat(*x) if not isinstance(x[0], list) else cs.blockcat(x),
        'zeros': cs.GenMX_zeros,
        'diag': lambda x: cs.diag(cs.DM(x)),
        'matmul': lambda x, y: cs.mtimes(x, y.reshape((x.shape[1],-1))),
        'concatenate': lambda x: cs.horzcat(*x) if x[0].shape[0] == 1 else cs.vertcat(*x),
        'norm': cs.norm_2, #lambda x: cs.sqrt(cs.mtimes(x.T, x)),
        'inv': cs.inv,
        'pinv': cs.pinv,

        'skew_symmetric': skew_symmetric_cs,
        'm2c': m2c_cs,
        'gvect': gvect_cs,
        'calculate_dcm': calculate_dcm_cs,
        'quaternion_to_dcm': quaternion_to_dcm_cs,
        'quaternion_to_angles': quaternion_to_angles_cs,
    }
}