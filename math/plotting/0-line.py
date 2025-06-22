#!user/bin/evn python 3
import matplotlib.pyplot as plt
import numpy as np

y=np.arange(0,11) **3

# plot y as a red solid line

plt.plot(np.arange(0,11),y, c="red",linestyle="-", linewidth=2)
plt.show()

