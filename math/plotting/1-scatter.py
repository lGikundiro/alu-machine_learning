#!user/bin/evn python 3
import matplotlib.pyplot as plt
import numpy as np

mean=[69,0]
cov=[[15,8], [8,15]]
np.random.seed(5)
x, y =np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# Plot x ↦ y as a scatter plot
plt.scatter(x,y,c="magenta")

#label the axes
plt.xlabel('Height (in)')
plt.ylabel('weight (lbs)')

#add the title
plt.title("Men's Height Vs Weight")

#show the plot
plt.show()


