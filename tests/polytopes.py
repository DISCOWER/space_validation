import pypolycontain as pp
import numpy as np


# range from 0 to 2 pi
theta = np.linspace(0, 2*np.pi, 100)

# plot 2*cos(theta) and 2*sin(theta)
x = 2 * np.abs(np.cos(theta))
y = 2 * np.abs(np.sin(theta))

import matplotlib.pyplot as plt

plt.plot(theta, x+y)
plt.axis('equal')
plt.xlabel('theta')
plt.ylabel('2*cos(theta) + 2*sin(theta)')
plt.show()
