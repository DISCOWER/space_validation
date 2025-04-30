import numpy as np
import matplotlib.pyplot as plt

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
    
    def is_inside(self, point:np.ndarray) -> bool:
        assert len(point) == self.dim, "Point must have the same dimension as the hyperrectangle."
        return np.all(point >= self.lower_bounds) and np.all(point <= self.upper_bounds)
    
    def plot(self, ax:plt.Axes, color='blue', alpha=0.5):
        rect = plt.Rectangle(self.lower_bounds[:2], self.size[0], self.size[1], 
                             alpha=alpha, color=color)
        ax.add_patch(rect)