import numpy as np
from helpers import HyperRectangle


class Robot:
    def __init__(self,n_x:int,n_u:int):
        self.n_x = n_x  # Number of state variables
        self.n_u = n_u  # Number of control inputs
        self.x = np.zeros((n_x,))
        self.u = np.zeros((n_u,))

        self.fx = None  # State dynamics function
        self.gx = None

    def set_state(self, x:np.ndarray):
        assert x.shape == (self.n_x,), f"State vector must be of shape ({self.n_x},)"
        self.x = x

    def set_control(self, u:np.ndarray):
        assert u.shape == (self.n_u,), f"Control vector must be of shape ({self.n_u},)"
        self.u = u

    def dynamics(self, u:np.ndarray)-> np.ndarray:
        assert u.shape == (self.n_u,), f"Control vector must be of shape ({self.n_u},)"
        self.u = u
        # Calculate the dynamics
        dx = self.fx(self.x) + self.gx(self.x)@self.u
        return dx
    
    def step(self, x=None, u=None, dt=0.1)-> np.ndarray:
        # Update the state using the dynamics
        if x is not None:
            self.set_state(x)
        if u is not None:
            self.set_control(u)
        # Calculate the dynamics
        dx = self.dynamics(u)
        self.x += dx * dt
        return self.x


class FreeFlyer(Robot):
    def __init__(self):
        super().__init__(n_x=4, n_u=2)
        # dynamics in the form: dx = f(x) + g(x)u
        self.fx = lambda x: np.array([[0,0,1,0],
                                      [0,0,0,1],
                                      [0,0,0,0],
                                      [0,0,0,0]])@x
        self.gx = lambda x: np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

        self.U = HyperRectangle(np.array([-1, -1]), np.array([1, 1]))


class BlueROV(Robot):
    def __init__(self):
        super().__init__(n_x=4, n_u=2)
        # dynamics in the form: dx = f(x) + g(x)u
        self.fx = lambda x: np.array([[0,0,1,0],
                                      [0,0,0,1],
                                      [0,0,-0.1,0],
                                      [0,0,0,-0.1]])@x
        self.gx = lambda x: np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

        self.U = HyperRectangle(np.array([-2, -2]), np.array([2, 2]))