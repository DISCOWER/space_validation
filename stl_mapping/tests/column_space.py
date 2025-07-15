import numpy as np

import sys 
sys.path.append("..")
from stl_mapping.Utilities.sets import Zonotope, exists_K, compute_K


B = np.array([[1, 0], [0, 1]])
C = np.array([[1, 1], [0, 1]])

U = Zonotope(x=np.zeros((2,)), Gdiag=np.array([1, 1]))
D = Zonotope(x=np.zeros((2,)), Gdiag=np.array([0.4, 0.4]))

K = compute_K(B, C)
print(f"K: {K}")
print(f"inv(K): {np.linalg.pinv(K)}")

# plot the zonotopes
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
U.plot(axs[0], color='blue', alpha=0.5, label='U')
D.plot(axs[0], color='red', alpha=0.5, label='D')
axs[0].set_xlim(-2, 2)
axs[0].set_ylim(-2, 2)
axs[0].set_title("D and U")
axs[0].set_aspect('equal', adjustable='box')
axs[0].legend()

# plot U and KD
KD = D.linear_transform(K)
U.plot(axs[1], color='blue', alpha=0.5, label='U')
KD.plot(axs[1], color='green', alpha=0.5, label='KD')
axs[1].set_xlim(-2, 2)
axs[1].set_ylim(-2, 2)
axs[1].set_title("KD and U")
axs[1].set_aspect('equal', adjustable='box')
axs[1].legend()

# plot BU and CD
BU = U.linear_transform(B)
CD = D.linear_transform(C)
BU.plot(axs[2], color='blue', alpha=0.5, label='BU')
CD.plot(axs[2], color='red', alpha=0.5, label='CD')
axs[2].set_xlim(-2, 2)
axs[2].set_ylim(-2, 2)
axs[2].set_title("CD and BU")
axs[2].set_aspect('equal', adjustable='box')
axs[2].legend()


plt.savefig('tests/figures/column_space.png')