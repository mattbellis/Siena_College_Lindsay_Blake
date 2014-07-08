import matplotlib.pylab as plt
import numpy as np

x = np.random.random(10)
y = np.random.random(10)

fig = plt.figure()
plt.plot(x,y,'ko',markersize=5)

x = np.linspace(0,6.28,1000)
y = np.sin(x)

fig = plt.figure()
plt.plot(x,y,'r-')

plt.show()
