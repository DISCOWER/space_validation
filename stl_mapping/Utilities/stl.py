import gurobipy as gp
from Utilities.sets import Polytope
import numpy as np

class OptProbItems:
    def __init__(self, x_vars:gp.Var, u_vars:gp.Var, times:np.ndarray):
        self.x_vars = x_vars
        self.u_vars = u_vars
        self.times = times


def in_interval(items, I, i):
    t = items.times[i]
    return I[0] <= t <= I[1]

class Pred():
    def __init__(self, type, I=[0,0], preds=[], dims=[]):
        self.type = type
        self.I = I          # Time interval
        self.preds = preds  # List of child predicates (e.g. \phi1 Until_I \phi2 -> preds=[\phi1, \phi2])
        self.rho = None     # Variable for the satisfaction degree
        self.rhos = None    # Variables for the satisfaction degree in MU operator
        self.area = None    # Area for MU operator

        # check that if preds[0] is a Polytope, dims is of same lenght as preds[0].H.shape[1]
        if len(preds) > 0 and isinstance(preds[0], Polytope):
            if len(dims) > preds[0].H.shape[1]:
                raise ValueError("dims must be of same length as preds[0].H.shape[1]")
        self.dims = dims    # Dimensions that need to be considered for the polytope inequalities

    def print(self):
        return f"{self.type}_[{self.I[0]},{self.I[1]}]"
    
class Spec():
    def __init__(self, phi, t0, tf):
        self.phi = phi
        self.t0 = t0
        self.tf = tf


def quant_parse_operator(opt:gp.Model,pred:Pred, items):
    print(f"\ntype: {pred.type}")
    try:
        for predi in pred.preds:
            print(f"type in loop: {predi.type}")
            quant_parse_operator(opt,predi, items)
    except:
        pass
        print("reached the end")

    if pred.type == "AND":
        quant_AND(opt,pred,items)

    elif pred.type == "OR":
        quant_OR(opt,pred,items)

    elif pred.type == "F":
        quant_EVENTUALLY(opt,pred,items)

    elif pred.type == "G":
        quant_ALWAYS(opt,pred,items)

    elif pred.type == "U":
        quant_UNTIL(opt,pred,items)

    elif pred.type == "MU":
        quant_MU(opt,pred,items)

    elif pred.type == "NEG":
        quant_NEG(opt,pred,items)

    else:
        ValueError(f"Unknown operator {pred.type}")


def quant_AND(opt:gp.Model, pred:Pred, items:OptProbItems):
    pred.rho = opt.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, 
                          name=f"rho_AND_{pred.print()}")
    try:
        rhos = [p.rho for p in pred.preds]
        opt.addConstr(pred.rho == gp.min_(rhos), name=f"rho_AND_{pred.print()}")
    except Exception as e:
        print(f"Error in quant_AND: {e}")
    opt.update()

def quant_OR(opt:gp.Model, pred:Pred, items:OptProbItems):
    pred.rho = opt.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, 
                          name=f"rho_OR_{pred.print()}")
    try:
        rhos = [p.rho for p in pred.preds]
        opt.addConstr(pred.rho == gp.max_(rhos), name=f"rho_OR_{pred.print()}")
    except Exception as e:
        print(f"Error in quant_OR: {e}")
    opt.update()

def quant_EVENTUALLY(opt:gp.Model, pred:Pred, items):
    pred.rho = opt.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, 
                          name=f"rho_F_{pred.print()}")
    try:
        rhos = []
        for p in pred.preds:
            for i in range(p.rhos.shape[0]):
                if in_interval(items, pred.I, i):
                    rhos.append(p.rhos[i])
        opt.addConstr(pred.rho == gp.max_(rhos), name=f"rho_F_{pred.print()}")
    except Exception as e:
        print(f"Error in quant_EVENTUALLY: {e}")
    opt.update()

def quant_ALWAYS(opt:gp.Model, pred:Pred, items:OptProbItems):
    pred.rho = opt.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, 
                          name=f"rho_G_{pred.print()}")
    try:
        rhos = []
        for p in pred.preds:
            for i in range(p.rhos.shape[0]):
                check = in_interval(items, pred.I, i)
                if check:
                    rhos.append(p.rhos[i])
        opt.addConstr(pred.rho == gp.min_(rhos), name=f"min_rhos_G_{pred.print()}")
    except Exception as e:
        print(f"Error in quant_ALWAYS: {e}")
    opt.update()

def quant_UNTIL(opt:gp.Model, pred:Pred, items:OptProbItems):
    pred.rho = opt.addVar(vtype=gp.GRB.CONTINUOUS, name=f"rho_U_{pred.print()}")
    try:
        rhos = []
        for i in range(len(pred.preds[0].rhos)):
            if in_interval(items, pred.I, i):
                rho = opt.addVar(vtype=gp.GRB.CONTINUOUS, name=f"rho_U_{pred.print()}_i{i}")
                rhos.append(rho)
                rho_phi1 = opt.addVar(vtype=gp.GRB.CONTINUOUS, name=f"rho_phi1_{pred.preds[0].print()}_i{i}")
                opt.addConstr(rho_phi1 == gp.min_([r for r in pred.preds[0].rhos[0:i+1]]))
                opt.addConstr(rho == gp.min_([pred.preds[1].rhos[i], rho_phi1]))
        opt.addConstr(pred.rho == gp.max_(rhos), name=f"rho_U_{pred.print()}")
    except Exception as e:
        print(f"Error in quant_UNTIL: {e}")

def quant_MU(opt:gp.Model, pred:Pred, items:OptProbItems):
    print(f"quant_MU: {pred.print()}")
    N = items.x_vars.shape[0]
    N_faces = pred.preds[0].N_faces
    dims = pred.dims

    pred.rhos = opt.addMVar((N,), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, 
                            name=f"rho_MU_{pred.print()}")
    pred.rho_faces = []
    try:
        for i in range(N):
            rho_faces = opt.addMVar((N_faces,), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                    name=f"rho_faces_MU_{pred.print()}_i{i}")
            pred.rho_faces.append(rho_faces)
            for face in range(N_faces):
                ineqs = pred.preds[0].H @ items.x_vars[i,dims] - pred.preds[0].b
                opt.addConstr(ineqs[face] == -rho_faces[face], name=f"rho_faces_MU_{pred.print()}_i{i}_face{face}")
            opt.addConstr(pred.rhos[i] == gp.min_([rf for rf in rho_faces]), name=f"rho_MU_{pred.print()}_i{i}")

    except Exception as e:
        print(f"Error in quant_MU: {e}")
    opt.update()

def quant_NEG(opt:gp.Model, pred:Pred, items:OptProbItems):
    print(f"quant_NEG: {pred.print()}")
    if pred.preds[0].type == "MU":
        pred.rhos = opt.addMVar((items.x_vars.shape[0],), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, 
                                name=f"rho_NEG_{pred.print()}")
        try:
            for i in range(items.x_vars.shape[0]):
                opt.addConstr(pred.rhos[i] == -pred.preds[0].rhos[i], name=f"rho_NEG_{pred.print()}_i{i}")
        except Exception as e:
            print(f"Error in quant_NEG: {e}")
    else:
        pred.rho = opt.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, 
                              name=f"rho_NEG_{pred.print()}")
        try:
            opt.addConstr(pred.rho == -pred.preds[0].rho, name=f"rho_NEG_{pred.print()}")
        except Exception as e:
            print(f"Error in quant_NEG: {e}")
    opt.update()
