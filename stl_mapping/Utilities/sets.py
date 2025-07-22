import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
import gurobipy as gp
import casadi as cs

# check if B is in the column space of C
# find a matrix K such that B = CK
def exists_K(B, C, tol=1e-8):
    # Check if each column of B lies in the column space of 
    for i in range(B.shape[1]):
        b = B[:, i]
        # Solve Ck ≈ b using least squares
        k, residuals, rank, s = np.linalg.lstsq(C, b, rcond=None)
        if np.linalg.norm(C @ k - b) > tol:
            return False
    return True

def compute_K(B, C, tol=1e-8):
    if exists_K(B, C, tol):
        return np.linalg.pinv(B)@C
    else:
        raise ValueError("No K exists such that C = BK")
    

class HyperRectangle():
    def __init__(self, lower_bounds:np.ndarray, upper_bounds:np.ndarray):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        assert len(self.lower_bounds) == len(self.upper_bounds), "Lower and upper bounds must have the same length."
        assert np.all(self.lower_bounds <= self.upper_bounds), "Lower bounds must be less than or equal to upper bounds."
        self.dim = len(self.lower_bounds)
        self.center = (self.lower_bounds + self.upper_bounds) / 2
        self.size = self.upper_bounds - self.lower_bounds
        self.volume = np.prod(self.size)

        # obtain the inequalities of the hyperrectangle in the form of Ax <= b
        self.A = np.array([[-1, 0],
                           [1, 0],
                           [0, -1],
                           [0, 1]])
        self.b = np.array([-self.lower_bounds[0],
                           self.upper_bounds[0],
                           -self.lower_bounds[1],
                           self.upper_bounds[1]])

    def sum(self, other):
        assert self.dim == other.dim, "Hyperrectangles must have the same dimension."
        new_lower_bounds = self.lower_bounds + other.lower_bounds
        new_upper_bounds = self.upper_bounds + other.upper_bounds
        return HyperRectangle(new_lower_bounds, new_upper_bounds)

    def subtract(self, other):
        assert self.dim == other.dim, "Hyperrectangles must have the same dimension."
        new_lower_bounds = self.lower_bounds - other.lower_bounds
        new_upper_bounds = self.upper_bounds - other.upper_bounds
        return HyperRectangle(new_lower_bounds, new_upper_bounds)
    
    def divide(self, other):
        assert self.dim == other.dim, "Hyperrectangles must have the same dimension."
        new_lower_bounds = self.lower_bounds / other.lower_bounds
        new_upper_bounds = self.upper_bounds / other.upper_bounds
        return new_lower_bounds
    
    def scalar_multiply(self, scalar:float):
        assert isinstance(scalar, (int, float)), "Scalar must be a number."
        new_lower_bounds = self.lower_bounds * scalar
        new_upper_bounds = self.upper_bounds * scalar
        return HyperRectangle(new_lower_bounds, new_upper_bounds)
    
    def is_inside(self, point:np.ndarray) -> bool:
        assert len(point) == self.dim, "Point must have the same dimension as the hyperrectangle."
        return np.all(point >= self.lower_bounds) and np.all(point <= self.upper_bounds)
    
    def plot(self, ax:plt.Axes, color='blue', alpha=0.5):
        rect = plt.Rectangle(self.lower_bounds[:2], self.size[0], self.size[1], 
                             alpha=alpha, fc=color, lw=1, ec='black')
        ax.add_patch(rect)

class Zonotope():
    def __init__(self, x=None, G=None, Gdiag=None):
        # assert not both G and Gdiag are not given
        assert not (G is None and Gdiag is None)

        self.x = x
        if G is not None:
            self.G = G
            self.Gdiag = np.diag(G)
        else:
            self.Gdiag = Gdiag
            try:
                self.G = np.diag(Gdiag)
            except:
                self.G = None

    def compute_vertices(self):
        V = self.x.copy()
        for iVertex in range(self.G.shape[1]):
            translation = self.G[:, iVertex]
            V = np.vstack([V + translation, V - translation])
            if iVertex > self.G.shape[0] -1:
                try:
                    V = V[ConvexHull(V).vertices]
                except:
                    raise ValueError("Could not compute convex hull")
        return V

    def linear_transform(self, A:np.ndarray):
        print(f"A: {A}, G: {self.G} \n A@G: {A @ self.G}")
        return Zonotope(x=A @ self.x, G=A @ self.G)

    def plot(self, ax:plt.Axes, color='r',alpha=0.2, label=""):
        vertices = self.compute_vertices()
        # order vertices according to their angle
        keys = np.arctan2(vertices[:,1]-self.x[1], vertices[:,0]-self.x[0])
        vertices = vertices[np.argsort(keys)]
        vertices = np.vstack((vertices, vertices[0]))  # close the polygon

        # and plot
        poly = patches.Polygon(vertices, closed=True, color=color, alpha=alpha)
        poly.set_label(label)
        ax.add_patch(poly)

class Polytope():
    def __init__(self, H:np.ndarray, b:np.ndarray):
        self.H = H
        self.b = b
        self.dim = H.shape[1]
        self.N_faces = H.shape[0]  # Number of faces in the area
    
    def __init__(self, rectangle:HyperRectangle):
        self.dim = len(rectangle.lower_bounds)
        self.N_faces = 2 * self.dim
        self.H = np.zeros((self.N_faces, self.dim))
        self.b = np.zeros(self.N_faces)
        for i in range(self.dim):
            self.H[i, i] = -1
            self.b[i] = -rectangle.lower_bounds[i]
            self.H[i + self.dim, i] = 1
            self.b[i + self.dim] = rectangle.upper_bounds[i]
        # print(f"\nPolytope created with {self.N_faces} faces and dimension {self.dim}")
        # print(f"H: {self.H}, b: {self.b}")

    def print(self):
        return f"Area_[faces={self.N_faces}]"

    def constrain_point_inside(self, prog:gp.Model, point:gp.Var):
        for i in range(self.N_faces):
            prog.addConstr(self.H[i, 0:self.dim] @ point[0:self.dim] <= self.b[i])

    def constrain_point_outside(self, prog:gp.Model, point:gp.Var):
        zs = prog.addVars(self.N_faces, vtype=gp.GRB.BINARY, name="zs")
        for i in range(self.N_faces):
            prog.addConstr(self.H[i, 0:self.dim] @ point[0:self.dim] >= self.b[i] - (1 - zs[i]) * 1e6)
        prog.addConstr(gp.quicksum([z for z in zs]) >= 1, name="at_least_one_face_outside")
    
    def plot(self, ax:plt.Axes, color='blue', alpha=0.5, label=""):
        # todo
        pass
